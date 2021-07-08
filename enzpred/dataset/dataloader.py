""" Class to load hts with convenient featurizations.

"""
import os
import logging
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset as TorchDataset

from sklearn import preprocessing
from enzpred.features import build_features, feature_selection
from enzpred.utils import parse_utils, file_utils

ChemFeaturizer = build_features.ChemFeaturizer
ProtFeaturizer = build_features.ProtFeaturizer
FeatureSelector = feature_selection.FeatureSelector

DATASET_ARGS = [
    (
        ["--hts-csv-file"],
        dict(action="store", help="Name of hts csv file input", default=None),
    ),
    (
        ["--ssa-ref-file"],
        dict(
            action="store",
            help=("Name of input structure references" "file to be used in rrsa"),
            default=None,
        ),
    ),
    (
        ["--substrate-cats-file"],
        dict(
            action="store",
            help=("Name of pickled file containing" " substrate categories "),
            default=None,
        ),
    ),
    (
        ["--substrate-cat"],
        dict(action="store", help=("Name of substrate cat to select"), default=None),
    ),
    (
        ["--debug-sample"],
        dict(
            default=1e-3,
            action="store",
            type=float,
            help="Fraction of the dataset to subsample for debugging",
        ),
    ),
]

DATASET_TYPES = ["HTSLoader"]


def get_dataset(dataset_type: str, **kwargs):
    """Get dataset mapping to avoid importing each class"""
    return {"HTSLoader": HTSData}[dataset_type](**kwargs)


def pad_then_to_tensor(
    l: list, PAD_TOKEN: int = 0, **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return tensor, length of each sequence"""

    # Don't pad and tensor if string, Data, etc.
    if type(l[0]) is str:
        return l, None
    elif type(l[0]) is np.ndarray and len(l[0].shape) > 1:
        # Only pad first dimension!
        seqlens = [len(i) for i in l]
        max_len = np.max(seqlens)

        padded_ar = np.array(
            [
                np.vstack([i, np.ones((max_len - len(i), i.shape[1])) * PAD_TOKEN])
                for i in l
            ]
        )
        # Switch dims to be batch, hidden, length
        return torch.Tensor(padded_ar).transpose(1, 2), torch.Tensor(np.array(seqlens))
    else:
        np_ar = np.vstack(l)
        return torch.from_numpy(np_ar).float(), (torch.ones(len(l)) * len(l[0])).long()


def rxn_collate(entries: List[pd.Series], PAD_TOKEN=0, train_mode=True) -> dict:
    """rxn_collate.

    Args:
        entries (List[pd.Series]): entries

    Returns:
        dict:
    """

    batch = {}
    if "rxn_features" in entries[0]:
        if isinstance(entries[0]["rxn_features"], tuple):
            rxn_features = [i["rxn_features"][0] for i in entries]
            features, _ = pad_then_to_tensor(rxn_features, PAD_TOKEN=PAD_TOKEN)
            batch["rxn_features"] = features

            # Global rxn features
            global_features = [i["rxn_features"][1] for i in entries]
            global_features, _ = pad_then_to_tensor(
                global_features, PAD_TOKEN=PAD_TOKEN
            )
            batch["concat_rdkit"] = global_features
        else:
            rxn_features = [i["rxn_features"] for i in entries]
            features, _ = pad_then_to_tensor(rxn_features, PAD_TOKEN=PAD_TOKEN)
            batch["rxn_features"] = features

    if "prot_features" in entries[0]:
        prot_features = [i["prot_features"] for i in entries]
        # Account for str featurizer
        if isinstance(prot_features[0], str):
            features, seqlens = prot_features, [len(i) for i in prot_features]
        else:
            features, seqlens = pad_then_to_tensor(prot_features, PAD_TOKEN=PAD_TOKEN)
        batch["prot_features"] = features
        batch["prot_seqlens"] = seqlens

    if train_mode:
        # Require labels
        labels = torch.Tensor([i["LABEL"] for i in entries])
        batch["labels"] = labels

    return batch


class BaseDataset(TorchDataset):
    """Base dataset used to give common properties to hts and rxndataset"""

    def __init__(self, data: pd.DataFrame, **kwargs):
        """__init__.

        Args:
            data (pd.DataFrame): data
            kwargs:
        """
        super(BaseDataset, self).__init__()
        self.data = data
        self.features = []
        cur_features = [j for j in ["prot_features", "rxn_features"] if j in self.data]
        self.features.extend(cur_features)

        # set length
        self._num_examples = len(self.data)
        self.scalers = {}

    def featurize_data(
        self,
        chem_featurizer_obj: Optional[ChemFeaturizer] = None,
        prot_featurizer_obj: Optional[ProtFeaturizer] = None,
        **kwargs,
    ):
        """featurize_data.

        Args:
            chem_featurizer_obj (Optional[ChemFeaturizer]): chem_featurizer_obj
            prot_featurizer_obj (Optional[ProtFeaturizer]): prot_featurizer_obj
            kwargs:
        """

        # Featurize the proteins
        if prot_featurizer_obj:
            logging.info("Featurizing proteins")
            self.data["prot_features"] = list(
                prot_featurizer_obj.set_featurize(self.data["SEQ"].values)
            )
            self.features.append("prot_features")

        # logging.info("Out of first protein featurize")

        # Featurize the compounds
        if chem_featurizer_obj:
            logging.info("Featurizing molecules")
            self.data["rxn_features"] = list(
                chem_featurizer_obj.set_featurize(self.data["SUBSTRATES"].values)
            )
            self.features.append("rxn_features")
        # logging.info("Out of first mol featurize")

    def select_features(
        self,
        column_name: str,
        feature_selector: Optional[FeatureSelector] = None,
        train: bool = False,
        **kwargs,
    ) -> Optional[int]:
        """select_features.

        Args:
            column_name (str) : "chem" or "prot" for which to select from
            feature_selector (Optional[FeatureSelector]): feature_selector
            train (bool): If true (default false), train and then select
            kwargs:
        Return:
            Optional[int]: New outdimension
        """

        def run_selector(df_list, y_vals, selector, train, **kwargs) -> List[np.array]:
            """Actually run the selector"""
            X = np.stack(df_list)
            # Check shape
            if train:
                selector.fit(X=X, y=y_vals)
            return list(selector.transform(X))

        y_vals = None
        if train:
            # use actual labels if HTS else use EC proxy for global
            if self.__class__ == HTSData:
                y_vals = self.get_labels()

        # Get the corresponding feature name
        feat_name = {"prot": "prot_features", "chem": "rxn_features"}.get(
            column_name, None
        )

        ret = None
        if feature_selector is None:
            logging.info(f"None feature selector for col {column_name}")
        elif feat_name and feat_name in self.data:
            new_feats = run_selector(
                self.data[feat_name], y_vals, feature_selector, train, **kwargs
            )
            self.data[feat_name] = new_feats
            ret = len(new_feats[0])

        else:
            logging.info(f"Unable to run feature selection for {column_name}")
        return ret

    def scale_features(
        self,
        column_name: str,
        scaler: preprocessing.StandardScaler,
        train: bool = False,
        **kwargs,
    ):
        """scale_features.

        Args:
            column_name (str) : "chem" or "prot" for which to scale
            train (bool): If true (default false), train and then select
            scaler (preprocessing.StandardScaler) : Scaler
            kwargs:
        Return:
            Optional[int]: New out-dimension
        """

        def run_scaler(df_list, scaler, train, **kwargs) -> List[np.array]:
            """Actually run the selector"""
            X = np.stack(df_list)
            # Check shape
            if train:
                scaler.fit(X=X)
            return list(scaler.transform(X))

        # Get the corresponding feature name
        feat_name = {"prot": "prot_features", "chem": "rxn_features"}.get(
            column_name, None
        )
        ret = None
        if feat_name and feat_name in self.data:
            new_feats = run_scaler(self.data[feat_name], scaler, train, **kwargs)
            self.data[feat_name] = new_feats
        else:
            logging.info(f"Unable to run feature selection for {column_name}")

    def scale_targets(
        self,
        column_name: str,
        scaler: preprocessing.StandardScaler,
        train: bool = False,
        **kwargs,
    ):
        """scale_features.

        Args:
            column_name (str) : "chem" or "prot" for which to scale
            train (bool): If true (default false), train and then select
            scaler (preprocessing.StandardScaler) : Scaler
            kwargs:
        Return:
            Optional[int]: New out-dimension
        """

        def run_scaler(df_list, scaler, train, **kwargs) -> List[np.array]:
            """Actually run the selector"""
            X = np.stack(df_list)
            # Check shape
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            if train:
                scaler.fit(X=X)

            return list(scaler.transform(X).flatten())

        targ_vals = self.data[column_name]
        new_vals = run_scaler(targ_vals, scaler, train, **kwargs)

        self.data[column_name] = new_vals

    def export_dataset(self, outfile: str) -> None:
        """export_dataset.

        Args:
            outfile (str): outfile

        Returns:
            None:
        """
        # Take all data except features..
        cols = [i for i in self.data.columns.values if "features" not in i]
        self.data[cols].to_csv(outfile)

    def add_pred_column(self, predictions: np.ndarray, col_name: str) -> None:
        """add_predictions.

        Add predicted values to the data

        Args:
            predictions (np.ndarray) : Predictions made for this data
            col_name (str): Column name
        """
        self.data[col_name] = predictions.tolist()

    def add_predictions(self, predictions: np.ndarray) -> None:
        """add_predictions.

        Add predicted values to the data

        Args:
            predictions (np.ndarray) : Predictions made for this data
        """
        self.add_pred_column(predictions, "predictions")

    def add_predictions_regr(
        self, predictions: np.ndarray, col_names: List[str]
    ) -> None:
        """add_predictions.

        Add predicted values to the data

        Args:
            predictions (np.ndarray) : Predictions made for this data
        """
        for column, col_name in enumerate(col_names):
            self.add_pred_column(predictions[:, column], col_name)

    def get_labels(self) -> np.array:
        """Get labels returning"""
        raise NotImplementedError("Not implemented")

    def get_feature_df(self):
        """Return a df containing all the features for the data.

        Use this method so no other functions interface directly with the
        pandas df as a property.
        """
        return self.data[self.features]

    def __len__(self):
        return self._num_examples

    def __getitem__(self, idx):

        if self.not_set:
            self._labels = self.data[self.label_columns].values
            self._prot_features = (
                self.data["prot_features"].values
                if "prot_features" in self.data.columns
                else None
            )
            self._rxn_features = (
                self.data["rxn_features"].values
                if "rxn_features" in self.data.columns
                else None
            )
            self.not_set = False

        entry = {}
        if self._prot_features is not None:
            entry["prot_features"] = self._prot_features[idx]
        if self._rxn_features is not None:
            entry["rxn_features"] = self._rxn_features[idx]
        label_vals = self._labels[idx].tolist()

        entry["LABEL"] = label_vals

        # if not 0 <= idx < self._num_examples:
        #    raise IndexError(idx)
        # entry = self.data.loc[idx]
        ## generate label values when we need them
        # label_val = entry[self.label_columns].values.tolist()
        # entry["LABEL"] = label_val

        return entry


##### Dense datasets #####


class HTSData(BaseDataset):
    """HTSData."""

    def __init__(
        self, data: pd.DataFrame, ssa_ref_file: Optional[str] = None, **kwargs
    ):

        """__init__.

        Hold basic object of sbustrates, seq, and ec_num

        Some of the super methods won't work

        Args:
            data (pd.DataFrame): data should contain cols "SUBSTRATES",
                "SEQ" and "EC_NUM"
            ssa_ref_file (Optional[str]): Name of reference file. Default None.
            kwargs:
        """
        super(HTSData, self).__init__(data, **kwargs)
        self.not_set = True
        self.label_columns = set(data.keys()).difference(
            ["SUBSTRATES", "SEQ", "rxn_features", "prot_features", "LABEL"]
        )

        self.label_columns = sorted(list(self.label_columns))
        self.labels_subsetted = False

        # self.data['LABEL'] = self.data[self.label_columns].values.tolist()
        self.ssa_ref_file = ssa_ref_file

    def get_ref_obj(
        self,
        prot_featurizer_: Optional[ProtFeaturizer] = None,
        align_dist: Optional[str] = None,
        **kwargs,
    ) -> Optional[Tuple]:
        """Get a protein reference from ssa ref
        Args:
            prot_featurizer_ (Optional[ProtFeaturizer]): Optional prot
                featurizer
            align_dist (Optional[str]): align dist type. If this is
                rssa_random, then replace with random embeddings
        Return:
            Tuple of residues and embedding

        """
        if self.ssa_ref_file is None or prot_featurizer_ is None:
            return None
        else:
            self.ref_seq, self.pool_residues = parse_utils.parse_ssa_reference(
                self.ssa_ref_file
            )
            self.pool_residues = sorted(self.pool_residues)
            self.ref_embedding = prot_featurizer_.set_featurize([self.ref_seq])[0]

            # Randomize
            if align_dist == "rssa_random":
                seqlen = np.shape(self.ref_embedding)[0]
                num_residues_chosen = len(self.pool_residues)
                new_positions = np.random.choice(
                    a=np.arange(seqlen), replace=False, size=num_residues_chosen
                )
                self.pool_residues = new_positions
                logging.info("Randomizing pool residues")

            return (self.pool_residues, self.ref_embedding)

    @classmethod
    def fuse_hts(cls, *args, **kwargs) -> object:
        """Fuse a list of hts datasets and return a fused instance"""
        fused_df = pd.concat([i.data for i in args]).reset_index(drop=True)
        new_data = HTSData(data=fused_df, **kwargs)
        return new_data

    def sort_by_seqs(self):
        self.data = self.data.sort_values(by="SEQ")
        return self

    def random_label_subset(self, new_cols=None, max_k=5):
        """Randomly choose a subset of the labels"""
        if new_cols is not None:
            self.labels_subsetted = True
            self.full_labels = self.label_columns
            self.label_columns = new_cols
        else:
            label_set = self.label_columns
            self.full_labels = label_set
            label_len = len(label_set)
            subset = np.random.choice(
                label_set, replace=False, size=min(max_k, label_len)
            )
            self.label_columns = subset
        self.not_set = True

    def reset_label_subset(self):
        """Return random label subset to normal"""

        if self.labels_subsetted:
            self.label_columns = self.full_labels

    def get_labels(self) -> np.array:
        """Get labels returning"""
        temp = self.data[self.label_columns].values
        return temp

    def num_tasks(self) -> np.array:
        """Get number of tasks"""
        return len(self.label_columns)

    def get_label_names(self) -> np.array:
        """Get number of tasks"""
        return self.label_columns

    def get_seqs(self):
        """Return a df containing all the features for the data.

        Use this method so no other functions interface directly with the
        pandas df as a property.
        """
        return self.data["SEQ"]


def filter_substrates(
    df, substrate_cats_file: Optional[str], substrate_cat: Optional[str], **kwargs
) -> pd.DataFrame:
    """filter_substrates.

    Given a dataframe, filter out all substrates that don't fall into
    substrate_cat as defined by the user

    Args:
        df (pd.DataFrame): Data to filter
        substrate_cats_file (Optional[str]): Name of pickle file containing a
            mapping of substrate category to substrates
        substrate_cat (Optional[str]): Name of substrate category to keep for
            analysis
        kwargs: Extra arguments

    Return:
        New, updated dataframe with substrates filtered
    """

    # Skip if no substrate file, no substrate cat, or no path exists
    if (
        substrate_cats_file is None
        or substrate_cat is None
        or not os.path.exists(substrate_cats_file)
    ):
        return df

    else:
        categories = file_utils.pickle_load(substrate_cats_file)
        if substrate_cat not in categories:
            logging.info(
                f"Could not find category {substrate_cat} in {substrate_cats_file}"
            )
            return df

        valid_subs = categories[substrate_cat]
        truth_ar = []
        prev_valid = len(pd.unique(df["SUBSTRATES"]))
        for val in df["SUBSTRATES"].values:
            truth_ar.append(val in valid_subs)

        new_df = df[np.array(truth_ar)].reset_index(drop=True)
        new_valid = len(pd.unique(df["SUBSTRATES"]))
        logging.info(
            f"After filtering for {substrate_cat}, {new_valid} "
            f"/ {prev_valid} subs left"
        )
        return new_df


def skip_col(
    vals: np.ndarray,
    regression: bool = False,
    max_imbalance: float = 0.9,
    min_size: int = 0,
    min_positives: int = 2,
    **kwargs,
) -> bool:
    """skip_col.

    Checks to see if the column fails the imbalance threshold or is below min
    size. Return true to skip the col if either of these are true.

    Args:
        vals (np.ndarray) : Array of values
        regression (bool) : If true, regression and ignore imbalance
        max_imbalance (float): Fraction of imbalance in data to be tolerated
        min_size (int): Minimum number of datapoints to be tolerated
        min_positives (int): If classification, minimimum num of positives
        kwargs: Extra arguments

    Return:
        True if col to be skipped
    """
    valid = True
    flattened_data = vals.flatten()
    # first purge all the nan value
    flattened_data = flattened_data[~np.isnan(flattened_data)]
    if not regression:
        num_positive = np.sum(flattened_data)
        frac_pos = num_positive / len(flattened_data)

        if frac_pos > max_imbalance or frac_pos < 1 - max_imbalance:
            valid = False

        if num_positive < min_positives:
            valid = False

    if valid and len(flattened_data) <= min_size:
        valid = False
    return not valid


def pivot_task_df(
    data: pd.DataFrame, pivot_task: Optional[str], skip_cols=False, **kwargs
) -> pd.DataFrame:
    """pivot_task_df.

    Reformat the data frame such that substrates or sequences become multitasks

    Args:
        data (pd.DataFrame): data
        pivot_task (Optional[str]): pivot_task
        skip_cols (bool): If False, don't skip any columns
            This is better for distance evaluation.
        kwargs:

    Returns:
        pd.DataFrame:
    """
    if pivot_task is None:
        return data

    index = "SEQ" if pivot_task == "SUBSTRATES" else "SUBSTRATES"
    data = data.pivot(index=index, columns=pivot_task).reset_index()
    data.columns = ["_".join([j for j in i if j != ""]) for i in data.keys()]
    # Replace the pivoted value with None for consistency
    data[pivot_task] = None
    # Now we should filter out of DF
    valid_cols = [index, pivot_task]
    for (col_name, col_data) in data.iteritems():
        if col_name == index or col_name == pivot_task:
            continue
        to_skip = skip_col(col_data.values, **kwargs)
        if to_skip and skip_cols:
            logging.info(f"Removing col {col_name}")
        else:
            valid_cols.append(col_name)
    logging.info(f"Num valid cols: {len(valid_cols) - 2}")
    data = data[valid_cols].reset_index(drop=True)
    return data


def unpivot_label(label: str, pivot_task: Optional[str], **kwargs) -> dict:
    """unpivot_label.

    Given a label, unpivot it given the format defined above

    Args:
        label (str): Pivoted label
        pivot_task (Optional[str]): pivot_task
        kwargs:

    Returns:
        dict
    """
    if pivot_task is None:
        return {}

    index = "SEQ" if pivot_task == "SUBSTRATES" else "SEQ"

    new_label, pivot_item = label.rsplit("_", 1)
    res_dict = {"label": new_label, pivot_task: pivot_item}
    return res_dict
