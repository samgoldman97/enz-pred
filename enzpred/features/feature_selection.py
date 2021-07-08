"""Feature selection wrapper. 
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn.decomposition import PCA


def empty_fn(*args, **kwargs):
    return None


FEATURE_SELECTION_TYPES = ["chi2", "variance", "pca", None]

FEATURE_SELECTION_ARGS = [
    (
        ["--num-k-best"],
        dict(
            action="store", type=int, default=30, help="Number of features to prune to"
        ),
    ),
    (
        ["--n-components"],
        dict(
            action="store",
            type=int,
            default=10,
            help="Number of principle components to store",
        ),
    ),
    (
        ["--prot-selector"],
        dict(
            action="store",
            type=str,
            default=None,
            choices=FEATURE_SELECTION_TYPES,
            help="Feature selector for proteins",
        ),
    ),
    (
        ["--chem-selector"],
        dict(
            action="store",
            type=str,
            default=None,
            choices=FEATURE_SELECTION_TYPES,
            help="Feature selector for chem repr",
        ),
    ),
    (
        ["--var-select-threshold"],
        dict(
            action="store",
            type=float,
            default=0.05,
            help="Variance for univariate variance selector. Higher is more stringent ",
        ),
    ),
]


def get_feature_selection(feature_selection: str, **kwargs):
    return {
        "chi2": Chi2,
        "variance": VarianceThresholdWrapper,
        "pca": PCADecomposition,
        None: empty_fn,
    }[feature_selection](**kwargs)


class FeatureSelector(ABC):
    """FeatureSelector."""

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs: kwargs
        """
        super().__init__()

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """fit.

        Args:
            X (np.ndarray): X
            y (np.ndarray): y
            kwargs: kwargs
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """transform.

        Args:
            X (np.ndarray): X
            kwargs: kwargs

        Returns:
            np.ndarray:
        """
        pass


class Univariate(FeatureSelector):
    """Univariate Feature selector."""

    def fit(self, X: np.ndarray, **kwargs):
        """fit.

        Args:
            X (np.ndarray): X
            kwargs : kwargs
        """
        self.model = self.model.fit(X)

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """transform.

        Args:
            X (np.ndarray): X
            kwargs: kwargs

        Returns:
            np.ndarray
        """
        return self.model.transform(X)


class Bivariate(FeatureSelector):
    """Bivariate Feature selector; depends on X and y."""

    def fit(self, X: np.ndarray, y, **kwargs):
        """fit.

        Args:
            X (np.ndarray): X
            y (np.ndarray): y
            kwargs : kwargs
        """
        self.model = self.model.fit(X, y)

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """transform.

        Args:
            X (np.ndarray): X
            kwargs: kwargs

        Returns:
            np.ndarray
        """
        return self.model.transform(X)


class Chi2(Bivariate):
    """Chi2."""

    def __init__(self, num_k_best: int, **kwargs):
        """__init__.

        Args:
            num_k_best (int): num_features
            kwargs: kwargs
        """

        super().__init__()
        self.k = num_k_best
        self.model = SelectKBest(chi2, self.k)


class VarianceThresholdWrapper(Univariate):
    """VarianceThreshold."""

    def __init__(self, var_select_threshold: float = 0.01, **kwargs):
        """__init__.

        Args:
            threshold (float): Variance to select away
            kwargs: kwargs
        """

        super().__init__()
        self.model = VarianceThreshold(threshold=var_select_threshold)


class PCADecomposition(Univariate):
    """VarianceThreshold."""

    def __init__(self, n_components: int = 10, **kwargs):
        """__init__.

        Args:
            n_components (int): Num principle components for pca
            kwargs: kwargs
        """

        super().__init__()
        self.model = PCA(n_components=n_components)
