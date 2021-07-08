"""Data splitter.
"""

import logging
import sys
import pandas as pd
import numpy as np
from typing import Tuple, Iterator, Optional
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from enzpred.dataset import dataloader
from enzpred.utils import file_utils


def get_splitter_dense(splitter_name: str, **kwargs):
    return {
        "random": RandomFolds,
        "kfold": KFoldSplitter,
        "loo-seq": LOOSeq,
        "loo-sub": LOOSub,
        "kfold-seq": KFoldSeq,
        "kfold-sub": KFoldSub,
        "single-sub": ProteinOnlySplitter,
        "single-prot": SubstrateOnlySplitter,
    }[splitter_name](**kwargs)


DENSE_SPLITTER_NAMES = [
    "random",
    "single-prot",
    "single-sub",
    "loo-seq",
    "loo-sub",
    "kfold-seq",
    "kfold-sub",
]

DENSE_SPLITTER_ARGS = [
    (
        ["--train-size"],
        dict(action="store", default=0.8, type=float, help="Size of train"),
    ),
    (["--val-size"], dict(action="store", default=0.1, type=float, help="Size of val")),
    (
        ["--test-size"],
        dict(action="store", default=0.1, type=float, help="Size of test"),
    ),
    (
        ["--count-positives"],
        dict(
            action="store_true",
            default=False,
            help="If true, stratify dataset to maintain correct fractions of true examples not absolute examples",
        ),
    ),
    (
        ["--num-folds"],
        dict(action="store", default=1, type=int, help="""Number of folds"""),
    ),
    (
        ["--num-kfold-trials"],
        dict(
            action="store",
            default=1,
            type=int,
            help="""Number of times to repeat k fold experiment""",
        ),
    ),
    (
        ["--split-groups-file"],
        dict(
            action="store",
            default=None,
            type=str,
            help="Name of pickle file mapping group names to group items in axis (e.g., map taxons to sequences)",
        ),
    ),
    (
        ["--max-imbalance"],
        dict(
            action="store",
            default=0.9,
            type=float,
            help="Maximum data imbalance. If the minor class has < max_imbalance points, filter the fold",
        ),
    ),
    (
        ["--no-loo-pool"],
        dict(
            action="store_true",
            default=False,
            help=(
                "By default, treat dense screens with leave one out pooling in"
                "splits. If this is set, don't use LOO pooling; compute"
                "metrics for each split individually"
            ),
        ),
    ),
    (
        ["--sub-split-type"],
        dict(
            action="store",
            default="loo",
            help="If using single task splitter, define how to split each single task",
            choices=["loo", "random"],
        ),
    ),
]


####### Generic Splitters #######


class Splitter(ABC):
    """Abstract class"""

    def __init__(self, train_size: float, val_size: float, test_size: float, **kwargs):
        """__init__.

        Args:
            train_size (float): train_size
            val_size (float): val_size
            test_size (float): test_size
            kwargs:
        """
        super(Splitter, self).__init__()
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        if not np.isclose(self.train_size + self.val_size + self.test_size, 1):
            raise ValueError(
                f"Train, val test {train_size, val_size, test_size} does not add to 1"
            )

    def split(
        self,
        data: Dataset,
        out: str,
        only_export_split_indices: bool = False,
        regression: bool = False,
        export_split_name: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """split.

        This abstracts away the retrieval of the split indices

        Args:
            data (Dataset): data
            export_split_name (str) : name of export file
            out (str): Name of out prefix
            only_export_split_indices (bool) : If true, exit program
                after saving splits
            regression (bool):
            kwargs:

        Returns:
            Tuple[Dataset, Dataset, Dataset]:
        """
        train_indices, val_indices, test_indices = self.get_split_indices(
            data, **kwargs
        )

        if export_split_name is not None:
            pickle_out = out + f"_{export_split_name}"
            logging.info(f"Pickling train indices to {pickle_out}")
            file_utils.pickle_obj(
                (train_indices, val_indices, test_indices), pickle_out
            )

        if only_export_split_indices:
            sys.exit("Exiting program early after saving train,val,test,indices")

        # data_objs = np.array([dict(i) for i in data])
        get_data_at_indices = lambda x: data.data.loc[x].reset_index(drop=True)

        train = get_data_at_indices(
            train_indices
        )  # pd.DataFrame(data_objs[train_indices].tolist())

        # Make val optional
        if len(val_indices) > 0:
            val = get_data_at_indices(val_indices)
        else:
            val = pd.DataFrame([])

        # Make test optional
        if len(test_indices) > 0:
            test = get_data_at_indices(test_indices)
        else:
            test = pd.DataFrame([])

        train.reset_index(drop=True, inplace=True)
        val.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)

        train_dataset = dataloader.get_dataset(data=train, **kwargs)
        val_dataset = dataloader.get_dataset(data=val, **kwargs)
        test_dataset = dataloader.get_dataset(data=test, **kwargs)

        if regression:
            # Log pos examples
            logging.info(
                f"examples: {len(data):,}| "
                f"examples in train: {len(train_dataset):,} | "
                f"examples in val: {len(val_dataset):,}| "
                f"examples in test: {len(test_dataset):,}"
            )
        else:
            data_labels = data.get_labels()
            train_labels = train_dataset.get_labels()
            val_labels = val_dataset.get_labels()
            test_labels = test_dataset.get_labels()
            logging.info(
                f"Total pos examples: {np.sum(data_labels):,} / {np.sum(~np.isnan(data_labels)):,}| "
                f"pos examples in train: {np.sum(train_labels):,} / {np.sum(~np.isnan(train_labels)):,} | "
                f"pos examples in val: {np.sum(val_labels):,} / {np.sum(~np.isnan(val_labels)):,}| "
                f"pos examples in test: {np.sum(test_labels):,} / {np.sum(~np.isnan(test_labels)):,}"
            )

        return train_dataset, val_dataset, test_dataset

    def get_split_indices(
        self, data: Dataset, **kwargs
    ) -> Tuple[np.array, np.array, np.array]:
        """get_split_indices.

        Args:
            data (Dataset): data
            kwargs:

        Returns:
            Tuple[np.array, np.array, np.array]:
        """
        raise NotImplementedError()


class IndexSplitter(Splitter):
    """Helper splitter to just generically split based on the index provided."""

    def __init__(self, **kwargs):
        """
        Args:
            split_indices_file (str): Name of split file
        """
        super(IndexSplitter, self).__init__(**kwargs)

    def get_split_indices(
        self, data, train, val, test, **kwargs
    ) -> Tuple[np.array, np.array, np.array]:
        """get_split_indices.

        Args:
            kwargs:

        Returns:
            Tuple[np.array, np.array, np.array]:
        """
        return (train, val, test)


class RandomSplitter(Splitter):
    """Randomly split"""

    def __init__(self, train_size: float, val_size: float, test_size: float, **kwargs):
        super(RandomSplitter, self).__init__(train_size, val_size, test_size, **kwargs)

    def get_split_indices(
        self, data: Dataset, **kwargs
    ) -> Tuple[np.array, np.array, np.array]:
        """get_split_indices.

        Args:
            data (Dataset): data
            kwargs:

        Returns:
            Tuple[np.array, np.array, np.array]:
        """
        # Unpack data set
        # Make sure to add index
        data_objs = []
        for index, i in enumerate(data):
            append_dict = dict(i)
            append_dict["index"] = index
            data_objs.append(append_dict)

        train, test_val = train_test_split(
            data_objs,
            train_size=self.train_size,
            test_size=(self.val_size + self.test_size),
            shuffle=True,
        )

        # Adjust for equzl portioning
        new_denom = self.val_size + self.test_size

        if self.val_size == 0:
            val = []
            test = test_val
        elif self.test_size == 0:
            test = []
            val = test_val
        else:
            val, test = train_test_split(
                test_val,
                train_size=self.val_size / new_denom,
                test_size=self.test_size / new_denom,
                shuffle=True,
            )
        # Convert to DF's
        train = pd.DataFrame(train).set_index("index")
        val = pd.DataFrame(val)
        if len(val) != 0:
            val = val.set_index("index")
        test = pd.DataFrame(test)
        if len(test) != 0:
            test = test.set_index("index")

        return train.index.values, val.index.values, test.index.values


####### Dense Splitters#######


class DenseSplitter(ABC):
    """For a dense splitter, this will return multiple folds"""

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs:
        """
        super(DenseSplitter, self).__init__()

    @abstractmethod
    def get_splits(self, data: Dataset, **kwargs) -> Iterator:
        """Return name, train, val, test"""
        raise NotImplementedError()


class KFoldSplitter(DenseSplitter):
    """Return k fold splits"""

    def __init__(self, no_loo_pool: bool = False, num_folds: int = 3, **kwargs):
        """__init__.

        Args:
            no_loo_pool (bool): If true, pool all items together. Treat them
                as separate metrics.
            num_folds (int): Number of k folds to use
            kwargs:
        """

        super(KFoldSplitter, self).__init__(**kwargs)
        self.index_splitter = IndexSplitter(**kwargs)
        self.loo_pool = not no_loo_pool
        self.num_folds = num_folds

    def get_splits(self, data: Dataset, **kwargs) -> Iterator:
        """get_splits.

        Args:
            data (Dataset): data
            kwargs:

        Returns:
            Iterator:
        """
        ### Assume for now the axis is
        # Turn this into groups
        # Split loo vals into num folds groups
        data_indices = np.arange(len(data))
        np.random.shuffle(data_indices)
        num_folds = min(self.num_folds, len(data_indices))
        folds = np.array_split(data_indices, num_folds)

        if num_folds < 2:
            assert "Fold splitter must have more than fold"

        loo_vals = {
            f"Fold_{fold_index}": set(group.tolist())
            for fold_index, group in enumerate(folds)
        }

        all_indices = set(range(len(data_indices)))

        for left_out_name, test_indices in loo_vals.items():
            logging.info(f"Leaving out {left_out_name}")

            train_superset_indices = np.array(
                list(all_indices.difference(test_indices))
            )
            test_indices = np.array(list(test_indices))
            # train_superset_indices = np.array([i for i in data_indices
            #                                   if i not in test_indices])
            np.random.shuffle(train_superset_indices)

            train_cutoff = int(
                (1 - self.index_splitter.val_size) * len(train_superset_indices)
            )

            train_indices = train_superset_indices[:train_cutoff]
            val_indices = train_superset_indices[train_cutoff:]
            train_val_test = self.index_splitter.split(
                data, train=train_indices, val=val_indices, test=test_indices, **kwargs
            )

            if self.loo_pool:
                fold_name = f"KFold"
            else:
                fold_name = f"{left_out_name}"
            yield (fold_name, *train_val_test)


class RandomFolds(DenseSplitter):
    """Return several splits of a dense splitter"""

    def __init__(
        self,
        train_size: float,
        val_size: float,
        test_size: float,
        num_folds: int = 1,
        **kwargs,
    ):
        """__init__.

        Args:
            train_size (float): train_size
            val_size (float): val_size
            test_size (float): test_size
            num_folds (int): num_folds
            kwargs:
        """

        super(RandomFolds, self).__init__(**kwargs)
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_splitter = RandomSplitter(
            self.train_size, self.val_size, self.test_size, **kwargs
        )
        self.num_folds = num_folds

    def get_splits(self, data: Dataset, **kwargs) -> Iterator:
        """get_splits.

        Args:
            data (Dataset): data
            kwargs:

        Returns:
            Iterator:
        """
        for split_num in range(self.num_folds):
            train, val, test = self.random_splitter.split(data, **kwargs)
            yield (split_num, train, val, test)


class LOOSuper(DenseSplitter):
    """Return several splits of a leave one out scheme"""

    def __init__(
        self,
        loo_axis: str = "SUBSTRATES",
        split_groups_file: str = None,
        no_loo_pool: bool = False,
        num_folds: int = 3,
        use_folds: bool = False,
        num_kfold_trials: int = 1,
        **kwargs,
    ):
        """__init__.

        Args:
            val_size (float): Fraction of the training set to siphon off into
                val
            loo_axis (str) : Name of the axis to do the leave one out over
            split_groups_file (str) : Name of the split groups file
            no_loo_pool (bool): If true, pool all items together. Treat them
                as separate metrics.
            num_folds (int): Number of k folds to use
            use_folds (bool): If true, use folds rather than holding out
                individual items
            num_kfold_trials (int): Number of times to repeat the kfold
                experiment
            kwargs:
        """

        super(LOOSuper, self).__init__(**kwargs)
        self.loo_axis = loo_axis
        self.split_groups = None
        if split_groups_file is not None:
            self.split_groups = file_utils.pickle_load(split_groups_file)
        self.index_splitter = IndexSplitter(**kwargs)
        self.loo_pool = not no_loo_pool
        self.use_folds = use_folds
        self.num_folds = num_folds
        self.num_kfold_trials = num_kfold_trials

    def get_splits(self, data: Dataset, **kwargs) -> Iterator:
        """get_splits.

        Args:
            data (Dataset): data
            kwargs:

        Returns:
            Iterator:
        """

        # Repeat num_kfold_trials_times
        for trial_num in range(self.num_kfold_trials):
            ### Assume for now the axis is
            # Turn this into groups
            if self.split_groups is not None:
                loo_vals = self.split_groups
            elif self.use_folds:
                # Split loo vals into num folds groups
                loo_vals = pd.unique(data.data[self.loo_axis])
                np.random.shuffle(loo_vals)
                folds = np.array_split(loo_vals, self.num_folds)
                if self.num_folds < 2:
                    assert "Fold splitter must have more than fold"
                loo_vals = {
                    f"Fold_{fold_index}": group.tolist()
                    for fold_index, group in enumerate(folds)
                }
            else:
                # Use LOO
                # Turn this into a mapping of group naem to held out values
                loo_vals = pd.unique(data.data[self.loo_axis])
                loo_vals = {i: [i] for i in loo_vals}

            for left_out_name, left_out_group in loo_vals.items():
                logging.info(f"Leaving out {self.loo_axis} value {left_out_name}")

                test_bools = np.array(
                    [j in left_out_group for j in data.data[self.loo_axis]]
                )
                test_indices = np.argwhere(test_bools).flatten()
                train_superset = ~test_bools
                train_superset_indices = np.argwhere(train_superset).flatten()

                #  LOO axis
                np.random.shuffle(train_superset_indices)
                train_cutoff = int(
                    (1 - self.index_splitter.val_size) * len(train_superset_indices)
                )
                train_indices = train_superset_indices[:train_cutoff]
                val_indices = train_superset_indices[train_cutoff:]
                train_val_test = self.index_splitter.split(
                    data,
                    train=train_indices,
                    val=val_indices,
                    test=test_indices,
                    **kwargs,
                )

                if self.loo_pool:
                    fold_name = self.loo_axis
                else:
                    fold_name = f"{left_out_name}"
                fold_name = f"{fold_name}_{trial_num}"
                yield (fold_name, *train_val_test)


class LOOSeq(LOOSuper):
    """Leave one protein out at a time"""

    def __init__(self, **kwargs):
        super(LOOSeq, self).__init__(loo_axis="SEQ", use_folds=False, **kwargs)


class LOOSub(LOOSuper):
    """Leave one protein out at a time"""

    def __init__(self, **kwargs):
        super(LOOSub, self).__init__(loo_axis="SUBSTRATES", use_folds=False, **kwargs)


class KFoldSeq(LOOSuper):
    """Leave one a group of seqs out at a time"""

    def __init__(self, **kwargs):
        super(KFoldSeq, self).__init__(loo_axis="SEQ", use_folds=True, **kwargs)


class KFoldSub(LOOSuper):
    """Leave a group of subs at a time"""

    def __init__(self, **kwargs):
        super(KFoldSub, self).__init__(loo_axis="SUBSTRATES", use_folds=True, **kwargs)


class SingleAxis(DenseSplitter):
    """The idea behind this splitter is to transform the CPI task in the dense
    grid to be about scoring single columns or single rows (i.e., many proteins
    against one substrate)"""

    def __init__(
        self,
        constant_axis: str = "SUBSTRATES",
        num_folds: int = 1,
        regression: bool = False,
        max_imbalance: float = 0.9,
        min_size: int = 8,
        no_loo_pool: bool = False,
        sub_split_type="loo",
        **kwargs,
    ):
        """__init__.

        Args:
            constant_axis (str): Which axis to hold constant. E.g., if SUBSTRATES,
                then we isolate each compound separately and only try to test
                ability to predict a new substrate value
            num_folds (int): How many folds / splits to perform for each
                isolated axis. E.g., if we predict along substrate axis, then
                for each individual protein, we will do num_folds for that. If
                15 proteins, num_folds = 3 would return 45 different data splits
            regression (bool): If true, then regression
            max_imbalance (float): Maximum fraction of positives or negatives
                allowed in a split to run it
            min_size (int): Minimum size of a dataset to let it go
            no_loo_pool (bool): If true, treat each random fold as its own
                split with a separate name. Otherwise, aggregate them.
            sub_split_type (str): Within each single axis, this keyword
                indicates how we should be performing each split. Choices: [loo,
                random]. Default "loo".
            kwargs:
        """
        super(SingleAxis, self).__init__(**kwargs)
        self.constant_axis = constant_axis
        self.index_splitter = IndexSplitter(**kwargs)
        if sub_split_type == "random":
            self.sub_splitter = RandomFolds(num_folds=num_folds, **kwargs)
        elif sub_split_type == "loo":
            # Do the opposite of whatever this splitter is
            loo_axis = "SUBSTRATES" if constant_axis == "SEQ" else "SEQ"
            self.sub_splitter = LOOSuper(loo_axis=loo_axis, **kwargs)
        else:
            raise NotImplementedError(f"Sub split type {sub_split_type}")

        self.max_imbalance = max_imbalance
        self.min_size = min_size
        self.loo_pool = not no_loo_pool
        self.regression = regression

    def is_valid_split(
        self, new_dataset: Dataset, constant_name: str = "", **kwargs
    ) -> bool:
        """is_valid_split.

        Returns true if the data should be run over, false if not

        Args:
            new_dataset (Dataset): New dataset we want to iterate ovr
            constant_name (str): Name of constant to be skipped
            kwargs:
        """

        # Check if the dataset is imbalanced
        labels = new_dataset.get_labels()
        if labels.shape[1] > 1:
            raise ValueError(
                "Unexpected multiple task dimension in single sub splitter"
            )
        to_skip = dataloader.skip_col(
            labels, self.regression, self.max_imbalance, self.min_size
        )
        logging.info(f"Skipping fold {constant_name} due to imbalance or low N")
        return not to_skip

    def get_splits(self, data: Dataset, **kwargs) -> Iterator:
        """get_splits.

        Args:
            data (Dataset): data
            kwargs:

        Returns:
            Iterator:
        """
        ### Assume for now the axis is
        # These are the values to keep constants
        constant_vals = pd.unique(data.data[self.constant_axis])
        constant_dict = {i: [i] for i in constant_vals}

        for constant_name, constant_group in constant_dict.items():
            logging.info(
                f"Isolating {self.constant_axis} value {constant_name} for prediction"
            )
            new_data_bools = np.array(
                [j in constant_group for j in data.data[self.constant_axis]]
            )

            new_data_indices = np.argwhere(new_data_bools).flatten()

            data_objs = np.array([dict(i) for i in data])
            new_data = pd.DataFrame(data_objs[new_data_indices].tolist())
            new_data.reset_index(drop=True, inplace=True)

            new_dataset = dataloader.get_dataset(data=new_data, **kwargs)

            # Make sure this is a valid dataset
            if not self.is_valid_split(new_dataset, constant_name, **kwargs):
                continue

            for split_num, train, val, test in self.sub_splitter.get_splits(
                new_dataset, **kwargs
            ):

                # Pool these over the same constant
                if self.loo_pool:
                    fold_name = constant_name
                else:
                    fold_name = f"{constant_name}_num_{split_num}"
                yield (fold_name, train, val, test)


class ProteinOnlySplitter(SingleAxis):
    """Test generalization across protein descriptors"""

    def __init__(self, **kwargs):
        """__init__."""

        super(ProteinOnlySplitter, self).__init__(constant_axis="SUBSTRATES", **kwargs)


class SubstrateOnlySplitter(SingleAxis):
    """Test generalization across substrate descriptors"""

    def __init__(self, **kwargs):
        """__init__."""
        super(SubstrateOnlySplitter, self).__init__(constant_axis="SEQ", **kwargs)
