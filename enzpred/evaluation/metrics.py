"""metrics.py
Class to hold all metrics

"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    silhouette_score,
    matthews_corrcoef,
)
from scipy.stats import spearmanr
import logging

METRIC_ARGS = []

## Metrics
def get_metrics(regression: bool, **kwargs):
    """Get different evaluation metrics"""
    if regression:
        return {
            "mae": MAE(**kwargs),
            "spearman": Spearman(**kwargs),
            "rmse": RMSE(**kwargs),
            "neg-spearman": NegSpearman(**kwargs),
        }

    else:
        return {
            "precision": PrecisionMetric(**kwargs),
            "recall": RecallMetric(**kwargs),
            "accuracy": AccuracyMetric(**kwargs),
            "mcc": MCCMetric(**kwargs),
            "f1": F1Metric(**kwargs),
            "auc-roc": AucRocMetric(**kwargs),
            "neg-auc-roc": AucRocMetric(negative=True, **kwargs),
            "neg-avg-pr": AvgPrecisionMetric(negative=True, **kwargs),
            "avg-pr": AvgPrecisionMetric(**kwargs),
        }


## Metrics
def get_clustering_metrics():
    """Get different evaluation metrics for clustering"""
    return {
        "mi_adjusted": adjusted_mutual_info_score,
        "rand_score_adjusted": adjusted_rand_score,
    }


def get_clustering_metrics_dataset(**kwargs):
    """Get different evaluation metrics for clustering.
    These take in the dataset and labels provided
    """

    return {
        "MSEClusteringLoss": MSEClusteringLoss(**kwargs),
        "MAEClusteringLoss": MAEClusteringLoss(**kwargs),
        "AvgClusteringLoss": AvgClusteringLoss(**kwargs),
        "silhouette_score": SilhouetteWrapper(**kwargs),
    }


METRIC_OPTIONS = [
    "precision",
    "recall",
    "accuracy",
    "f1",
    "mcc",
    "auc-roc",
    "avg-pr",
    "rmse",
    "mae",
]


class ClusteringMetricDataset(object):
    """Base class for metrics on clustering results that take in as input the
    actual output and a set of labels.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.metric = None

    def __call__(self, X: np.ndarray, labels: np.ndarray) -> float:
        """__call__.

        Args:
            X (np.ndarray): numpy 2d array
            labels (np.ndarray): Numpy array of labels

        Return:
            float score

        """
        pass


class MSEClusteringLoss(ClusteringMetricDataset):
    def __init__(self, **kwargs):
        """__init__."""
        super().__init__(**kwargs)

    def __call__(
        self, X: np.ndarray, labels: np.ndarray, mean_pool: bool = True
    ) -> float:
        """__call__.

        Args:
            X (np.ndarray): numpy 2d array
            labels (np.ndarray): Numpy array of labels

        Return:
            float score
        """

        unique_labels = np.unique(labels)
        l2_dists = []
        for label in unique_labels:
            indices = labels == label
            centroid = np.nanmean(X[indices], axis=0)

            # Now do k means loss

            dists = centroid[None, :] - X[indices]
            # Use euclidean NAN taken from sklearn
            squared_sum = np.nansum(np.square(dists), axis=-1)
            num_present = np.sum(~np.isnan(dists), axis=-1)
            num_total = dists.shape[-1]
            true_dist_mat = np.sqrt(num_total / num_present * squared_sum)

            l2_dists.extend(true_dist_mat.tolist())

        if mean_pool:
            return np.nanmean(l2_dists)
        else:
            return l2_dists


class MAEClusteringLoss(ClusteringMetricDataset):
    def __init__(self, **kwargs):
        """__init__."""
        super().__init__(**kwargs)

    def __call__(self, X: np.ndarray, labels: np.ndarray, mean_pool=True) -> float:
        """__call__.

        Args:
            X (np.ndarray): numpy 2d array
            labels (np.ndarray): Numpy array of labels

        Return:
            float score
        """
        unique_labels = np.unique(labels)
        l1_dists = np.zeros(labels.shape)
        for label in unique_labels:
            indices = labels == label
            centroid = np.nanmean(X[indices], axis=0)

            # Now do k means loss

            dists = centroid[None, :] - X[indices]
            # Use manhattan NAN taken from sklearn
            squared_sum = np.nansum(np.abs(dists), axis=-1)
            num_present = np.sum(~np.isnan(dists), axis=-1)
            num_total = dists.shape[-1]

            # Remove sqrt
            true_dist_mat = num_total / num_present * squared_sum

            l1_dists[indices] = true_dist_mat

            # .extend(true_dist_mat.tolist())

        if mean_pool:
            return np.nanmean(l1_dists)
        else:
            return l1_dists


class AvgClusteringLoss(ClusteringMetricDataset):
    """Compute the average distance from each point
    to the rest of the points its clustered with."""

    def __init__(self, **kwargs):
        """__init__."""
        super().__init__(**kwargs)

    def __call__(self, X: np.ndarray, labels: np.ndarray, mean_pool=True) -> float:
        """__call__.

        Args:
            X (np.ndarray): numpy 2d array
            labels (np.ndarray): Numpy array of labels

        Return:
            float score
        """
        unique_labels = np.unique(labels)
        l1_dists = np.zeros(labels.shape)
        for label in unique_labels:
            indices = labels == label

            this_cluster = X[indices]
            dists = this_cluster[:, None, :] - this_cluster[None, :, :]

            # Compute safe manhattan distance norm on the -1 axis
            abs_sum = np.nansum(np.abs(dists), axis=-1)
            num_present = np.sum(~np.isnan(dists), axis=-1)
            num_total = dists.shape[-1]

            true_dist_mat = num_total / num_present * abs_sum

            # Now take an average over all distances INSIDE the cluster
            true_dist_mat = np.nanmean(true_dist_mat, -1)

            l1_dists[indices] = true_dist_mat

        if mean_pool:
            return np.nanmean(l1_dists)
        else:
            return l1_dists


class SilhouetteWrapper(ClusteringMetricDataset):
    def __init__(self, **kwargs):
        """__init__."""
        super().__init__(**kwargs)

    def __call__(self, X: np.ndarray, labels: np.ndarray) -> float:
        """__call__.

        Args:
            X (np.ndarray): numpy 2d array
            labels (np.ndarray): Numpy array of labels

        Return:
            float score
        """
        if np.any(np.isnan(X)):
            return None
        else:
            return silhouette_score(X, labels)


class MetricBase(object):
    """Base class for metrics"""

    def __init__(self, **kwargs):
        super().__init__()
        self.metric = None

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Run metric forwards"""

        # Remove nan
        non_nan_vals = ~np.isnan(y_true)
        y_true_ = y_true[non_nan_vals]
        y_pred_ = y_pred[non_nan_vals]

        return self.metric(y_true_, y_pred_)

    def check_class(self, y_true: np.array) -> bool:
        """Return true if all members of y true are of same type"""
        y_true_identical = all(x == y_true[0] for x in y_true)
        return y_true_identical

    def threshold(self, ar: np.array, thresh: int = 0.5) -> np.array:
        """Threshold the array"""
        ar = ar.copy()
        ar[ar >= thresh] = 1
        ar[ar < thresh] = 0
        return ar


class MAE(MetricBase):
    """Mean average error"""

    def __init__(self, **kwargs):
        pass

    def metric(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))


class RMSE(MetricBase):
    """Root mean squared error"""

    def __init__(self, **kwargs):
        pass

    def metric(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_true - y_pred)))


class Spearman(MetricBase):
    """Root mean squared error"""

    def __init__(self, **kwargs):
        pass

    def metric(self, y_true, y_pred):
        return spearmanr(y_true, y_pred)[0]


class NegSpearman(MetricBase):
    """Root mean squared error"""

    def __init__(self, **kwargs):
        pass

    def metric(self, y_true, y_pred):
        return -spearmanr(y_true, y_pred)[0]


class AvgPrecisionMetric(MetricBase):
    def __init__(self, **kwargs):
        super(AvgPrecisionMetric, self).__init__(**kwargs)
        self.metric = average_precision_score


class AvgPrecisionMetric(MetricBase):
    def __init__(self, negative=False, **kwargs):
        super(AvgPrecisionMetric, self).__init__(**kwargs)
        self.metric = average_precision_score
        self.negative = -1 if negative else 1

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        return self.negative * super(AvgPrecisionMetric, self).__call__(y_true, y_pred)


class AucRocMetric(MetricBase):
    def __init__(self, negative=False, **kwargs):
        super(AucRocMetric, self).__init__(**kwargs)
        self.metric = roc_auc_score
        self.negative = -1 if negative else 1

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        if self.check_class(y_true):
            logging.warning(
                f"Unable to call metric {self.__class__} due to all items having same class"
            )
            return None
        else:
            return self.negative * super(AucRocMetric, self).__call__(y_true, y_pred)


class BinaryMetric(MetricBase):
    """Binary metric"""

    def __init__(self, **kwargs):
        super(BinaryMetric, self).__init__(**kwargs)
        self.metric = None

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """ """
        predictions = self.threshold(y_pred)
        return self.metric(y_true, predictions)


class PrecisionMetric(BinaryMetric):
    def __init__(self, **kwargs):
        super(PrecisionMetric, self).__init__(**kwargs)
        self.metric = precision_score


class RecallMetric(BinaryMetric):
    def __init__(self, **kwargs):
        super(RecallMetric, self).__init__(**kwargs)
        self.metric = recall_score


class AccuracyMetric(BinaryMetric):
    def __init__(self, **kwargs):
        super(AccuracyMetric, self).__init__(**kwargs)
        self.metric = accuracy_score


class MCCMetric(BinaryMetric):
    def __init__(self, **kwargs):
        super(MCCMetric, self).__init__(**kwargs)
        self.metric = matthews_corrcoef


class F1Metric(BinaryMetric):
    def __init__(self, **kwargs):
        super(F1Metric, self).__init__(**kwargs)
        self.metric = f1_score
