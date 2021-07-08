"""Class to hold models to operate on  
"""
from typing import Tuple, Optional
import itertools
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data.dataset import Dataset as TorchDataset

from sklearn.decomposition import PCA

from enzpred.models import sklearn_models, torch_models, distance
from enzpred.dataset import dataloader
from enzpred.utils import file_utils

# record model types
MODEL_TYPES = [
    "ffn",
    "ffndot",
    "ffnsingle",
    "random",
    "rf",
    "constant",
    "gp",
    "linear",
    "knn",
    "outterffn",
    "outterlinear",
]

# Get model args from sklearn models and torch models in case we use any
MODEL_ARGS = [
    (
        ["--batch-size"],
        dict(action="store", type=int, default=64, help="Batch size for training"),
    ),
    (
        ["--knn-uniform"],
        dict(
            action="store_true",
            default=False,
            help="If true, use uniform weights for KNN. Else, weight by distance",
        ),
    ),
    (
        ["--epochs"],
        dict(action="store", type=int, default=5, help="Batch size for training"),
    ),
    (
        ["--learning-rate"],
        dict(action="store", type=float, default=1e-3, help="Learning rate for adam"),
    ),
    (
        ["--gp-implementation"],
        dict(
            action="store",
            type=str,
            default="sklearn",
            choices=["sklearn"],
            help="Which gaussian process model to use",
        ),
    ),
    (
        ["--deep-ensemble-num"],
        dict(
            action="store",
            type=int,
            default=1,
            help="""Number of ensembles to use; only works with deep models""",
        ),
    ),
    (
        ["--seq-dist-type"],
        dict(
            action="store",
            type=str,
            default=None,
            choices=distance.SEQ_CHOICES,
            help="""Chocie of sequence distance for knn """,
        ),
    ),
    (
        ["--sub-dist-type"],
        dict(
            action="store",
            type=str,
            default=None,
            choices=distance.SUBSTRATE_CHOICES,
            help="""Choice of substrate distance for knn """,
        ),
    ),
    (
        ["--concat-val"],
        dict(
            action="store_true",
            default=False,
            help="If true, concatenate the validation set to teh train set in training loop",
        ),
    ),
]
MODEL_ARGS.extend(torch_models.MODEL_ARGS)
MODEL_ARGS.extend(sklearn_models.MODEL_ARGS)
MODEL_ARGS.extend(distance.MODEL_ARGS)


def get_model(model_type: str, **kwargs):
    """Clas to hold the code to get a model"""
    return {
        "ffn": FFNModel,
        "ffndot": FFNModelDotProd,
        "ffnsingle": FFNSingleTask,
        "random": RandomGuessModel,
        "constant": ConstantGuessModel,
        "rf": RandomForestModel,
        "gp": GaussianProcessModel,
        "linear": LinearModel,
        "knn": NearestNeighborModel,
        "outterlinear": OutterLinear,
        "outterffn": OutterFFN,
    }[model_type](**kwargs)


class RxnModel(object):
    """RxnModel.

    Super class to hold all rxn models

    """

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs: kwargs
        """
        super(RxnModel, self).__init__()

    def train(self, train_dataset: TorchDataset, val_dataset: TorchDataset, **kwargs):
        """fit.

        Fit the model on the torch datasets

        Args:
            train_dataset (TorchDataset): train_dataset
            val_dataset (TorchDataset): val_dataset
            kwargs:
        """
        raise NotImplementedError

    def predict(self, test_dataset: TorchDataset, **kwargs) -> Tuple[np.array, dict]:
        """predict.

        Args:
            test_dataset (TorchDataset): test_dataset
            kwargs:

        Returns:
            Tuple[np.arary, dict]: 1d array of probabilities of class 1; dict
                contains any auxilary prediction info
        """

        raise NotImplementedError

    def get_full_feature_set(
        self, dataset: dataloader.BaseDataset, **kwargs
    ) -> np.array:
        """get_full_feature_set.

        Args:
            dataset (TorchDataset): dataset
            kwargs:

        Returns:
            np.array:
        """

        def check_shape(ar: np.array):
            """Make sure this could be converted effectively to an ar"""
            if not len(ar.shape) == 2:
                raise ValueError(f"Unexpected error converting {ar} to np.array")

        dataset_data = dataset.get_feature_df()
        feature_ar = []

        if "rxn_features" in dataset_data:
            rxn_feats = np.stack(dataset_data["rxn_features"].values, axis=0)
            check_shape(rxn_feats)
            feature_ar.append(rxn_feats)

        if "prot_features" in dataset_data:
            prot_feats = np.stack(dataset_data["prot_features"].values, axis=0)
            check_shape(prot_feats)
            feature_ar.append(prot_feats)

        if len(feature_ar) > 0:
            concat_ar = np.concatenate(feature_ar, axis=1)
        else:
            concat_ar = []
        # logging.info(f"Shape of concatenated features {concat_ar.shape}")
        return concat_ar

    def export_model_stats(self, outprefix: str, **kwargs) -> None:
        """export_model_stats.

        Export model stats. This is specific to each model.
        E.g. Curves durinig training

        Args:
            outprefix (str) : Prefix of output to save stats to
            kwargs
        """
        return

    def load_state(self, save_file: str = None, **kwargs) -> None:
        """load_state.

        Args:
            save_file (str): save_file
            kwargs:

        Returns:
            None:
        """
        return

    def export_state(self, out_file: str, **kwargs) -> None:
        """export_state.

        Args:
            save_file (str): save_file
            kwargs:

        Returns:
            None:
        """
        pass


class SklearnModelWrapper(RxnModel):
    """SklearnModelWrapper."""

    def __init__(
        self, regression: bool, num_tasks: int = 1, concat_val: bool = False, **kwargs
    ):
        """__init__.

        Args:
            regression (bool): If true, run regressoin
            num_tasks (int): Used for regression to set num of regressors
                use multiple regressors bc sklearn can't handle NaN well
            concat_val (bool): If true concatenate validation to train when
                training
            kwargs: kwargs
        """
        super(SklearnModelWrapper, self).__init__(**kwargs)
        self.regression = regression
        self.tasks = num_tasks
        self.concat_val = concat_val

        # Define a model
        self.model = None

    def train(self, train_dataset: TorchDataset, val_dataset: TorchDataset, **kwargs):
        """fit.

        Fit the model on the torch datasets

        Args:
            train_dataset (TorchDataset): train_dataset
            val_dataset (TorchDataset): val_dataset
            kwargs:
        """
        input_data = train_dataset
        if self.concat_val and len(val_dataset) > 0:
            input_data = dataloader.HTSData.fuse_hts(train_dataset, val_dataset)

        X = self.get_full_feature_set(input_data, **kwargs)
        labels = input_data.get_labels()
        self._train(X, labels, **kwargs)

    def _train(self, X: np.ndarray, labels: np.ndarray, **kwargs):
        """_train.
        Internal training method that uses the feature set and label set,
        rather than the torch dataset.

        Args:
            X (np.ndarray): Python array of feature set
            labels (np.ndarray): Target values
            kwargs
        """
        for col in range(self.tasks):
            indices = ~np.isnan(labels[:, col])
            self.model[col].fit(X=X[indices], y=labels[indices, col], **kwargs)

    def _predict(self, X: np.ndarray, **kwargs):
        """_predict.

        Internal training method that uses the feature set and label set,
        rather than the torch dataset.

        Args:
            X (np.ndarray): Python array of feature set
            kwargs
        Return: predictions
        """
        preds = []
        for col in range(self.tasks):
            preds.append(self.model[col].predict(X, **kwargs))
        predictions = np.column_stack(preds)
        return predictions

    def predict(self, test_dataset: TorchDataset, **kwargs) -> Tuple[np.array, dict]:
        """predict.

        Args:
            test_dataset (TorchDataset): test_dataset
            kwargs:

        Returns:
            Tuple[np.arary, dict]: Get auxilary predictions
        """
        X = self.get_full_feature_set(test_dataset, **kwargs)
        predictions = self._predict(X, **kwargs)
        return predictions, {}


class RandomForestModel(SklearnModelWrapper):
    """RandomForestModel."""

    def __init__(self, **kwargs):
        """__init__.

        Args:
            regression (bool): If true, run regression
            num_tasks (int): Used for regression to set num of regressors
                use multiple regressors bc sklearn can't handle NaN well
            kwargs: kwargs
        """
        super(RandomForestModel, self).__init__(**kwargs)
        if self.regression:
            self.model = [
                sklearn_models.RandomForestRegressor(**kwargs)
                for _ in range(self.tasks)
            ]
        else:
            self.model = [
                sklearn_models.RandomForestClassifier(**kwargs)
                for _ in range(self.tasks)
            ]


class NearestNeighborModel(SklearnModelWrapper):
    """NearestNeighborModel."""

    def __init__(
        self,
        seq_dist_type: Optional[str],
        sub_dist_type: Optional[str],
        n_neighbors: int,
        knn_uniform: bool = False,
        **kwargs,
    ):
        """__init__.

        Args:
            regression (bool): If true, run regression
            num_tasks (int): Used for regression to set num of regressors
                use multiple regressors bc sklearn can't handle NaN well
            seq_dist_type (Optional[str]): Type of sequence distance
            sub_dist_type (Optional[str]): Type of substrate distance
            n_neighbors (int): Num of nearest neighbors
            knn_uniform (bool): If true, use unifrom weights
            concat_val (bool): If true, concatenate the validation set
            kwargs: kwargs
        """
        super(NearestNeighborModel, self).__init__(**kwargs)

        weights = "uniform" if knn_uniform else "distance"
        if self.regression:
            self.model = [
                sklearn_models.KNNRegressor(
                    n_neighbors=n_neighbors, weights=weights, **kwargs
                )
                for _ in range(self.tasks)
            ]
        else:
            self.model = [
                sklearn_models.KNNClassifier(
                    n_neighbors=n_neighbors, weights=weights, **kwargs
                )
                for _ in range(self.tasks)
            ]
        self.ref_data = None

        # Get sub and seq distances
        self.seq_dist_type, self.sub_dist_type = sub_dist_type, seq_dist_type
        self.seq_dist = distance.get_seq_dist(seq_dist_type, **kwargs)
        self.sub_dist = distance.get_sub_dist(sub_dist_type, **kwargs)
        self.n_neighbors = n_neighbors

    def compute_distance_matrix(
        self, reference_dataset: TorchDataset, new_dataset: TorchDataset, **kwargs
    ) -> np.ndarray:
        """compute_distance_matrix.

        Args:
            reference_dataset (TorchDataset): reference_dataset
            new_dataset (TorchDataset): new_dataset
            kwargs:

        Returns:
            np.ndarray: Distance matrix with shape len(new_dataset) x
                len(old_datset)
        """

        seq_dist_mat = None
        sub_dist_mat = None

        # If both of them, then use rankings!
        if self.seq_dist and self.sub_dist:
            seq_dist_mat = self.seq_dist.rank_sparse_dist(
                reference_dataset, new_dataset, top_n=self.n_neighbors + 2, **kwargs
            )
            sub_dist_mat = self.sub_dist.rank_sparse_dist(
                reference_dataset, new_dataset, top_n=self.n_neighbors + 2, **kwargs
            )

        elif self.seq_dist:
            seq_dist_mat = self.seq_dist.dist(
                reference_dataset, new_dataset, top_n=self.n_neighbors, **kwargs
            )
        elif self.sub_dist:
            sub_dist_mat = self.sub_dist.dist(
                reference_dataset, new_dataset, top_n=self.n_neighbors, **kwargs
            )
        else:
            raise RuntimeError()

        merged_mat = self.merge_mats(seq_dist_mat, sub_dist_mat, **kwargs)

        return merged_mat

    def merge_mats(
        self, seq_mat: Optional[np.ndarray], sub_mat: Optional[np.ndarray], **kwargs
    ) -> np.ndarray:
        """Strategy for combining the seq dist and sub dist matrices"""

        if seq_mat is None and sub_mat is None:
            raise RuntimeError("Found two none matrices")
        elif seq_mat is not None and sub_mat is None:
            return seq_mat
        elif seq_mat is None and sub_mat is not None:
            return sub_mat
        else:
            # Add them both up and assume they are ranks!
            # For now, just make these standard normal distributions.
            # Note: This is naive
            # row_norm_seq = (seq_mat - np.mean(seq_mat, 1) ) / np.std(seq_mat, 1)
            # row_norm_sub = (sub_mat- np.mean(sub_mat, 1) ) / np.std(sub_mat, 1)
            return seq_mat + sub_mat

    def _train(self, X: np.ndarray, labels: np.ndarray, **kwargs):
        """_train.

        Args:
            X (np.ndarray): Input pairwise distance array
            labels (np.ndarray): Corresponding labels

        """
        # Hold the valid indices for each model
        # Remember, some models will have gaps and we should subset these
        self.model_indices = []
        for col in range(self.tasks):
            indices = ~np.isnan(labels[:, col])
            self.model_indices.append(indices)
            X_temp = X[indices, :][:, indices]
            self.model[col].fit(X=X_temp, y=labels[indices, col], **kwargs)

    def train(self, train_dataset: TorchDataset, val_dataset: TorchDataset, **kwargs):
        """fit.

        Fit the model on the torch datasets

        Args:
            train_dataset (TorchDataset): train_dataset
            val_dataset (TorchDataset): val_dataset
            kwargs:
        """

        self.ref_data = train_dataset

        if self.concat_val and len(val_dataset) > 0:
            self.ref_data = dataloader.HTSData.fuse_hts(train_dataset, val_dataset)

        # Compute distance matrix
        X = self.compute_distance_matrix(self.ref_data, self.ref_data, **kwargs)
        labels = self.ref_data.get_labels()
        self._train(X, labels, **kwargs)

    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """_predict.

        Internal predict method that takes as input a pairwise dist matrix from
        new items to the current X.

        Args:
            X (np.ndarray): Distances to every item in train

        Return:
            np.ndarray: Predictions
        """
        # Repeat on each column individually
        preds = []
        for col in range(self.tasks):
            indices = self.model_indices[col]
            X_temp = X[:, indices]
            preds.append(self.model[col].predict(X_temp, **kwargs))
        predictions = np.column_stack(preds)
        return predictions

    def predict(
        self, test_dataset: TorchDataset, return_pairwise: bool = False, **kwargs
    ) -> Tuple[np.array, dict]:
        """predict.

        Args:
            test_dataset (TorchDataset): test_dataset
            return_pairwise (bool): If true, return pairwise interactions
            kwargs:

        Returns:
            Tuple[np.arary, dict]: Get auxilary predictions
        """
        if self.ref_data is None:
            raise RuntimeError("Trying to use KNN model before fitting")

        # Compute distance matrix
        X = self.compute_distance_matrix(self.ref_data, test_dataset, **kwargs)

        aux = {}
        if return_pairwise:
            X_aux = self.compute_distance_matrix(test_dataset, test_dataset, **kwargs)
            pair_indices_list = []
            embed_dists = []
            for i, j in itertools.combinations(np.arange(len(test_dataset)), 2):
                if i != j:
                    pair_indices_list.append((i, j))
                    embed_dists.append(X_aux[i, j])

            aux["correlations"] = {
                "pair_indices": pair_indices_list,
                "embed_dists": embed_dists,
            }

        predictions = self._predict(X, **kwargs)

        return predictions, aux


class LinearModel(SklearnModelWrapper):
    """LinearModel."""

    def __init__(self, **kwargs):
        """__init__.

        Args:
            regression (bool): If true, run regression
            num_tasks (int): Used for regression to set num of regressors
                use multiple regressors bc sklearn can't handle NaN well
            kwargs: kwargs
        """
        super(LinearModel, self).__init__(**kwargs)
        if self.regression:
            self.model = [
                sklearn_models.RidgeRegression(**kwargs) for _ in range(self.tasks)
            ]
        else:

            self.model = [
                sklearn_models.LogisticRegression(**kwargs) for _ in range(self.tasks)
            ]


class RandomGuessModel(RxnModel):
    """RandomGuessModel."""

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs: kwargs
        """
        super(RandomGuessModel, self).__init__(**kwargs)
        self.model = sklearn_models.RandomGuess(**kwargs)

    def train(self, train_dataset: TorchDataset, val_dataset: TorchDataset, **kwargs):
        """fit.

        Fit the model on the torch datasets

        Args:
            train_dataset (TorchDataset): train_dataset
            val_dataset (TorchDataset): val_dataset
            kwargs:
        """
        labels = train_dataset.get_labels()
        self.model.fit(X=[], y=labels, **kwargs)

    def predict(self, test_dataset: TorchDataset, **kwargs) -> Tuple[np.array, dict]:
        """predict.

        Args:
            test_dataset (TorchDataset): test_dataset
            kwargs:

        Returns:
            Tuple[np.arary, dict]:
        """
        # Don't bother getting the feature array for this
        num_preds = len(test_dataset)
        predictions = self.model.predict(np.arange(num_preds), **kwargs)
        return predictions, {}


class ConstantGuessModel(RxnModel):
    """ConstantGuessModel."""

    def __init__(self, regression: bool, **kwargs):
        """__init__.

        Args:
            regression (bool): If true, guess based on EC averages for each
                value
            kwargs: kwargs
        """
        super(ConstantGuessModel, self).__init__(**kwargs)
        self.model = sklearn_models.ConstantGuess(**kwargs)
        self.regression = regression

    def train(self, train_dataset: TorchDataset, val_dataset: TorchDataset, **kwargs):
        """fit.

        Fit the model on the torch datasets

        Args:
            train_dataset (TorchDataset): train_dataset
            val_dataset (TorchDataset): val_dataset
            kwargs:
        """
        labels = train_dataset.get_labels()
        if self.regression:
            # Get the EC label in each class
            ec_nums = train_dataset.get_ec_nums()
            labels = train_dataset.get_labels()
            # Get all freq dicts
            freq_dict = defaultdict(lambda: [])
            for ec_num, label in zip(ec_nums, labels):
                freq_dict[ec_num].append(label)

            # By default return overall mean of dataset
            overall_mean = np.nanmean(labels, 0)
            self.freq_dict = defaultdict(lambda: overall_mean.copy())
            for ec_num, label in freq_dict.items():
                self.freq_dict[ec_num] = np.nanmean(label, 0)
                for col in range(overall_mean.shape[0]):
                    # Replace NaN with global mean for that col
                    if np.isnan(self.freq_dict[ec_num][col]):
                        self.freq_dict[ec_num][col] = overall_mean[col]

        else:
            self.model.fit(X=[], y=labels, **kwargs)

    def predict(self, test_dataset: TorchDataset, **kwargs) -> Tuple[np.array, dict]:
        """predict.

        Args:
            test_dataset (TorchDataset): test_dataset
            kwargs:

        Returns:
            Tuple[np.arary, dict]:
        """
        # Don't bother getting the feature array for this
        if self.regression:
            # Get averages per class!
            ret_vals = []
            for ec in test_dataset.get_ec_nums():
                ret_vals.append(self.freq_dict[ec])
            return np.vstack(ret_vals), {}

        else:
            num_preds = len(test_dataset)
            predictions = (
                np.ones(num_preds) * self.model.freq
            )  # predict(np.arange(num_preds), **kwargs)
            return predictions, {}


class TorchWrapperModel(RxnModel):
    """TorchWrapperModel.

    Wrapper for the torch_models classes
    """

    def __init__(self, deep_ensemble_num: int, **kwargs):
        """__init__.

        Args:
            deep_ensemble_num (int) : Num of ensembles
            in_features (int): in_features
            layers (int): layers
            hidden_size (int): hidden_size
            kwargs: kwargs
        """
        super(TorchWrapperModel, self).__init__(**kwargs)
        self.models = [None for _ in range(deep_ensemble_num)]
        self.train_curves = [defaultdict(lambda: []) for j in range(deep_ensemble_num)]
        self.val_curves = [defaultdict(lambda: []) for j in range(deep_ensemble_num)]
        self.deep_ensemble_num = deep_ensemble_num

    def get_dataloader(self, dataset, train_=True, **kwargs):
        """Get train dataloader"""

        return torch_models.get_dataloader(dataset, shuffle=train_, **kwargs)

    def train(self, train_dataset: TorchDataset, val_dataset: TorchDataset, **kwargs):
        """fit.

        Fit the model on the torch datasets

        Args:
            train_dataset (TorchDataset): train_dataset
            val_dataset (TorchDataset): val_dataset
            kwargs:
        """
        self.train_loader = self.get_dataloader(train_dataset, train_=True, **kwargs)
        self.val_loader = self.get_dataloader(val_dataset, train_=True, **kwargs)

        for index, model in enumerate(self.models):
            train_results = torch_models.train_model(
                model, self.train_loader, self.val_loader, **kwargs
            )
            self.models[index] = train_results["best_model"]
            self.train_curves[index] = train_results["train_losses"]
            self.val_curves[index] = train_results["val_losses"]

    def predict(
        self,
        test_dataset: TorchDataset,
        evaluate_aux_preds: bool = False,
        export_uncertainty: bool = False,
        **kwargs,
    ) -> Tuple[np.array, dict]:
        """predict.

        Args:
            test_dataset (TorchDataset): test_dataset
            evaluate_aux_preds (bool) : If true, return the aux predictions
                from the model
            export_uncertainty (bool): If true, also export uncertainty
            kwargs:

        Returns:
            Tuple[np.arary, dict]:
        """
        loader = torch_models.get_dataloader(
            test_dataset, shuffle=False, train_mode=False, **kwargs
        )
        loader = self.get_dataloader(
            test_dataset, train_=False, train_mode=False, **kwargs
        )

        predictions = []
        aux_preds = []
        for index, model in enumerate(self.models):
            prediction, aux_pred = torch_models.get_predictions(
                model, loader, get_aux_preds=evaluate_aux_preds, **kwargs
            )
            predictions.append(prediction)
            aux_preds.append(aux_pred)

        # TODO: report all aux preds not just first of ensemble
        predictions_ = np.stack(predictions, axis=0)
        predictions = np.mean(predictions_, axis=0)

        if export_uncertainty:
            uncertainty = np.std(predictions_, axis=0)
            predictions = (predictions, uncertainty)

        return (predictions, aux_preds[0])

    def export_model_stats(self, outprefix: str, **kwargs) -> None:
        """export_model_stats.

        Export model stats. This is specific to each model.
        E.g. Curves durinig training

        Args:
            outprefix (str) : Prefix of output to save stats to
            kwargs
        """
        import matplotlib.pyplot as plt

        for index in range(self.deep_ensemble_num):
            pickled_file = outprefix + f"_train_val_curves_{index}.pkl"
            plot_file = outprefix + f"_train_val_curves_{index}.png"
            file_utils.pickle_obj(
                (self.train_curves[index], self.val_curves[index]), pickled_file
            )

            # Plot curves
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 5))
            for curve, ax, title, xlabel in zip(
                [self.train_curves[index], self.val_curves[index]],
                [ax1, ax2],
                ["Train loss", "Val loss"],
                ["Batch Number", "Epoch Number"],
            ):
                for label, y_vals in curve.items():
                    ax.plot(y_vals, label=label)

                ax.set_xlabel(xlabel)
                ax.set_ylabel("Loss Value")
                ax.set_title(title)
                ax.legend()
                ax.legend()

            fig.savefig(plot_file, bbox_inches="tight")

    def load_state(self, save_file: str = None, **kwargs) -> None:
        """load_state.

        Args:
            save_file (str): save_file
            kwargs:

        Returns:
            None:
        """

        state_dicts = torch.load(save_file)
        # Load file!
        for index, model in enumerate(self.models):
            state_dict = state_dicts[index]
            model.load_state_dict(state_dict)

    def export_state(self, out_file: str, **kwargs) -> None:
        """load_state.

        Args:
            save_file (str): save_file
            kwargs:

        Returns:
            None:
        """
        for index in range(self.deep_ensemble_num):
            self.models[index] = self.models[index].cpu()

        # Load file!
        torch.save([model.state_dict() for model in self.models], out_file)


class FFNModel(TorchWrapperModel):
    """FFNModel."""

    def __init__(
        self, in_features: int, layers: int = 3, hidden_size: int = 100, **kwargs
    ):
        """__init__.

        Args:
            in_features (int): in_features
            layers (int): layers
            hidden_size (int): hidden_size
            kwargs: kwargs
        """
        super(FFNModel, self).__init__(**kwargs)
        for index in range(self.deep_ensemble_num):
            self.models[index] = torch_models.SimpleFFN(
                in_features, layers, hidden_size, interaction_layer="concat", **kwargs
            )


class FFNSingleTask(TorchWrapperModel):
    """FFNSingleTask."""

    def __init__(
        self, in_features: int, layers: int = 3, hidden_size: int = 100, **kwargs
    ):
        """__init__.

        Args:
            in_features (int): in_features
            layers (int): layers
            hidden_size (int): hidden_size
            kwargs: kwargs
        """
        super(FFNSingleTask, self).__init__(**kwargs)
        for index in range(self.deep_ensemble_num):
            self.models[index] = torch_models.FFNNoSharing(
                in_features, layers, hidden_size, **kwargs
            )


class FFNModelDotProd(TorchWrapperModel):
    """FFNModel."""

    def __init__(
        self, in_features: int, layers: int = 3, hidden_size: int = 100, **kwargs
    ):
        """__init__.

        Args:
            in_features (int): in_features
            layers (int): layers
            hidden_size (int): hidden_size
            kwargs: kwargs
        """
        super(FFNModelDotProd, self).__init__(**kwargs)
        for index in range(self.deep_ensemble_num):
            self.models[index] = torch_models.SimpleFFN(
                in_features, layers, hidden_size, interaction_layer="dot", **kwargs
            )


class GaussianProcessModel(RxnModel):
    """ """

    def __init__(
        self, regression: bool, gp_implementation: str, num_tasks: int = 1, **kwargs
    ):
        """__init__.

        Args:
            gp_implementation (str): Which package to use to build these
            num_tasks (int) : Number of tasks
            kwargs: kwargs
        """
        super(GaussianProcessModel, self).__init__(**kwargs)
        self.regression = regression
        self.gp_implementation = gp_implementation
        self.model = None
        self.num_tasks = num_tasks
        self.tasks = self.num_tasks

    def train(self, train_dataset: TorchDataset, val_dataset: TorchDataset, **kwargs):
        """fit.

        Fit the model on the torch datasets

        Args:
            train_dataset (TorchDataset): train_dataset
            val_dataset (TorchDataset): val_dataset
            gp_cluster_ec_level (int): Level at which to cluster ecs
            kwargs:
        """
        self.model = []
        train_labels = train_dataset.get_labels()
        val_labels = val_dataset.get_labels()

        X = self.get_full_feature_set(train_dataset, **kwargs)
        if self.gp_implementation == "sklearn":
            """For sklearn, learn a list of models for each cluster to acct for multitask"""
            self.model = []
            for col in range(self.tasks):
                model_ = (
                    sklearn_models.GPRegresion(**kwargs)
                    if self.regression
                    else sklearn_models.GPClassification(**kwargs)
                )
                indices = ~np.isnan(train_labels[:, col])
                model_.fit(X=X[indices], y=train_labels[indices, col], **kwargs)
                self.model.append(model_)

    def predict(self, test_dataset: TorchDataset, **kwargs) -> Tuple[np.array, dict]:
        """predict.

        Args:
            test_dataset (TorchDataset): test_dataset
            kwargs:

        Returns:
            np.array:
        """
        if self.model is None:
            raise ValueError("Must train network before running predictions")

        # Do the predictions in groups for each model
        output_predictions = np.zeros((len(test_dataset), self.tasks))
        if self.gp_implementation == "sklearn":
            X_test = self.get_full_feature_set(test_dataset, **kwargs)
            preds = []
            for col in range(self.tasks):
                preds.append(self.model[col].predict(X_test, **kwargs))
            predictions = np.column_stack(preds)

        else:
            raise ValueError(f"No gp impelmentation: {self.gp_implementation}")

        return predictions, {}


############ Developing outter product based models #############
class OutterFeaturizer:
    def get_full_feature_set(
        self, dataset: dataloader.BaseDataset, **kwargs
    ) -> np.array:
        """get_full_feature_set.

        Args:
            dataset (TorchDataset): dataset
            kwargs:

        Returns:
            np.array:
        """

        gap_num = 20
        chem_feats_end = 1024
        cov_lim = 0.7
        num_aa = 21
        chem_feat_std_thresh = 0.4
        prot_feat_std_thresh = 0.4
        combo_percentile = 0

        def check_shape(ar: np.array):
            """Make sure this could be converted effectively to an ar"""
            if not len(ar.shape) == 2:
                raise ValueError(f"Unexpected error converting {ar} to np.array")

        dataset_data = dataset.get_feature_df()
        feature_ar = []

        if "rxn_features" in dataset_data:
            rxn_feats = np.stack(dataset_data["rxn_features"].values, axis=0)
            if not self.trained:
                # Now get chem featurizers
                self.valid_chem_cols = np.std(rxn_feats, 0) > chem_feat_std_thresh

            chem_feats = rxn_feats[:, self.valid_chem_cols]

            check_shape(chem_feats)
            feature_ar.append(chem_feats)

        if "prot_features" in dataset_data:
            prot_feats = np.stack(dataset_data["prot_features"].values, axis=0)
            if not self.trained:
                self.ident_converter = np.eye(num_aa)

                # process prot features
                coverage = 1 - np.mean(prot_feats == gap_num, 0)
                self.valid_prot_cols = coverage > cov_lim
                updated_prot_feats = prot_feats[:, self.valid_prot_cols]

                num_examples = updated_prot_feats.shape[0]
                prot_feats = self.ident_converter[updated_prot_feats].reshape(
                    num_examples, -1
                )

                self.prot_onehot_valid = np.std(prot_feats, 0) > prot_feat_std_thresh
                prot_feats = prot_feats[:, self.prot_onehot_valid]
            else:
                updated_prot_feats = prot_feats[:, self.valid_prot_cols]
                num_examples = updated_prot_feats.shape[0]
                prot_feats = self.ident_converter[updated_prot_feats].reshape(
                    num_examples, -1
                )
                prot_feats = prot_feats[:, self.prot_onehot_valid]

            check_shape(prot_feats)
            feature_ar.append(prot_feats)

        if len(feature_ar) == 1:
            dense_mat = np.concatenate(feature_ar, axis=1)
        else:
            # Try to combine with not concatenation
            dense_mat = np.einsum("xy, xz->xyz", prot_feats, chem_feats).reshape(
                num_examples, -1
            )
            if not self.trained:
                std = dense_mat.std(0)
                percentile_cutoff = np.percentile(std, combo_percentile)
                # self.combo_valid = std > percentile_cutoff
                self.pca = PCA(n_components=100)
                self.pca.fit(dense_mat)

            dense_mat = self.pca.transform(dense_mat)
            # dense_mat = dense_mat[:, self.combo_valid]

        return dense_mat

    def get_dataloader(self, dataset, train_=True, **kwargs):
        """Get train dataloader"""

        class OutterDataset(torch.utils.data.Dataset):

            ## Create a dataloader
            def __init__(self, feats, labels):
                self.labels = labels
                self.feats = feats

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return dict(x=self.feats[idx], labels=self.labels[idx])

        feat_set = self.get_full_feature_set(dataset, **kwargs)

        labels = dataset.get_labels()

        dataset_ = OutterDataset(feat_set, labels.astype(float))

        loader = torch.utils.data.DataLoader(
            dataset_, num_workers=0, shuffle=train_, batch_size=kwargs.get("batch_size")
        )

        return loader


class OutterLinear(OutterFeaturizer, LinearModel):
    """Outter product between morgan fp and MSA and featurizer."""

    def __init__(self, **kwargs):
        """__init__."""
        # RandomForestModel.__init__(self,**kwargs)
        super(OutterLinear, self).__init__(**kwargs)
        self.trained = False

    def train(self, *args, **kwargs):
        """Call super train and then just set a flag to indicate trained"""

        super(OutterLinear, self).train(*args, **kwargs)
        self.trained = True


class OutterFFN(OutterFeaturizer, TorchWrapperModel):
    """Outter product between morgan fp and MSA and featurizer."""

    def __init__(
        self, in_features: int, layers: int = 3, hidden_size: int = 100, **kwargs
    ):
        """__init__.

        Args:
            in_features (int): in_features
            layers (int): layers
            hidden_size (int): hidden_size
            kwargs: kwargs
        """
        super(OutterFFN, self).__init__(**kwargs)
        self.trained = False

    def train(self, train_dataset: TorchDataset, val_dataset: TorchDataset, **kwargs):
        """fit.

        Fit the model on the torch datasets

        Args:
            train_dataset (TorchDataset): train_dataset
            val_dataset (TorchDataset): val_dataset
            kwargs:
        """
        self.train_loader = self.get_dataloader(train_dataset, train_=True, **kwargs)
        self.trained = True
        self.val_loader = self.get_dataloader(val_dataset, train_=True, **kwargs)
        in_features = {"joint_feature_len": next(iter(self.train_loader))["x"].shape[1]}

        for index in range(self.deep_ensemble_num):
            self.models[index] = torch_models.SimpleFFN(
                in_features,
                # layers = kwargs.get("layers"),
                # hidden_size = kwargs.get("hidden_size"),
                interaction_layer="none",
                **kwargs,
            )
        for index, model in enumerate(self.models):
            train_results = torch_models.train_model(
                model, self.train_loader, self.val_loader, **kwargs
            )
            self.models[index] = train_results["best_model"]
            self.train_curves[index] = train_results["train_losses"]
            self.val_curves[index] = train_results["val_losses"]
