"""Class to hold regression based, top layer models from sklearn.
"""
from abc import ABC, abstractmethod
import numpy as np
from sklearn import neighbors, linear_model, ensemble, gaussian_process
from sklearn.svm import SVC

MODEL_ARGS = [
    (
        ["--max-depth"],
        dict(action="store", type=int, default=8, help="max depth of rf tree"),
    ),
    (
        ["--n-estimators"],
        dict(action="store", type=int, default=100, help="num estimators for rf tree"),
    ),
    (
        ["--n-neighbors"],
        dict(action="store", type=int, default=5, help="num nearest neighbors for knn"),
    ),
    (
        ["--solver"],
        dict(action="store", type=str, default="lbfgs", help="Name of solver to use"),
    ),
    (
        ["--alpha"],
        dict(action="store", type=float, default=1, help="Strength of regularizer"),
    ),
    (
        ["--no-class-weight"],
        dict(
            action="store_true",
            default=False,
            help="if true, do not use class weight for logistic regr",
        ),
    ),
]

# MODEL_TYPES = ["LogisticRegression", "RandomGuess", "ConstantGuess","SVM"]


def get_model(model_type: str, **kwargs):
    """Clas to hold the code to get a model"""
    return {
        "LogisticRegression": LogisticRegression,
        "RandomGuess": RandomGuess,
        "ConstantGuess": ConstantGuess,
        "SVM": SVM,
        "RidgeRegression": RidgeRegressoin,
    }[model_type](**kwargs)


class SKLearnModel(ABC):
    """SKLearnModel."""

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs: kwargs
        """
        super(SKLearnModel, self).__init__()

    @abstractmethod
    def fit(self, X, y, **kwargs):
        """fit.

        Args:
            X: X
            y: y
            kwargs: kwargs
        """
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        """predict.

        Args:
            X: X
            kwargs: kwargs
        """
        pass


class RidgeRegression(SKLearnModel):
    """LogisticRegression."""

    def __init__(self, alpha: int = 1, **kwargs):
        """__init__.

        Args:
            kwargs: kwargs
        """
        super(RidgeRegression, self).__init__(**kwargs)
        self.model = linear_model.Ridge(alpha=alpha)

    def fit(self, X, y, **kwargs):
        """fit.

        Args:
            X: X
            y: y
            kwargs: kwargs
        """
        self.model.fit(X, y.squeeze())

    def predict(self, X, **kwargs) -> np.ndarray:
        """predict.

        Args:
            X: X
            kwargs: kwargs
        """
        return self.model.predict(X)


class LogisticRegression(SKLearnModel):
    """LogisticRegression."""

    def __init__(self, alpha: int = 1, no_class_weight: bool = False, **kwargs):
        """__init__.

        Args:
            kwargs: kwargs
        """
        super(LogisticRegression, self).__init__(**kwargs)
        self.C = 1.0 / alpha
        class_weight = None if no_class_weight else "balanced"
        self.model = linear_model.LogisticRegression(
            C=self.C, class_weight=class_weight
        )
        self.use_model = True

    def fit(self, X, y, **kwargs):
        """fit.

        For this model, make sure that X is a pairwise distance matrix in the
        fit loop

        Args:
            X: X
            y: y
            kwargs: kwargs
        """

        if len(np.unique(y)) == 1:
            self.use_model = False
            self.model_out = list(np.unique(y))[0]
        else:
            self.model.fit(X, y.squeeze())

    def predict(self, X, **kwargs) -> np.ndarray:
        """predict.

        Args:
            X: X
            kwargs: kwargs
        """
        # Note, this assumes that we have binary inputs trained
        # If we have 1 class, let's only return that single class...
        if self.use_model:
            return self.model.predict_proba(X)[:, 1]
        else:
            # Return the only class from the train set
            output = np.ones(len(X)).reshape(-1, 1)
            output = output * self.model_out
            return output


class RandomForestClassifier(SKLearnModel):
    """RandomForestClassifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        no_class_weight: bool = False,
        **kwargs
    ):
        """__init__.

        Args:
            n_estimators (int): number of estimators
            max_depth = None
            kwargs: kwargs
        """
        super(RandomForestClassifier, self).__init__(**kwargs)
        class_weight = None if no_class_weight else "balanced"
        self.model = ensemble.RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, class_weight=class_weight
        )

    def fit(self, X, y, **kwargs):
        """fit.

        Args:
            X: X
            y: y
            kwargs: kwargs
        """

        if len(np.unique(y)) == 1:
            self.use_model = False
            self.model_out = list(np.unique(y))[0]
        else:
            self.model.fit(X, y.squeeze())

    def predict(self, X, **kwargs) -> np.ndarray:
        """predict.

        Args:
            X: X
            kwargs: kwargs
        """
        return self.model.predict_proba(X)[:, 1]


class RandomForestRegressor(SKLearnModel):
    """RandomForestRegressor."""

    def __init__(self, n_estimators: int = 100, max_depth: int = None, **kwargs):
        """__init__.

        Args:
            max_depth (int): max depth
            kwargs: kwargs
        """
        super(RandomForestRegressor, self).__init__(**kwargs)
        self.model = ensemble.RandomForestRegressor(
            max_depth=max_depth, n_estimators=n_estimators
        )

    def fit(self, X, y, **kwargs):
        """fit.

        Args:
            X: X
            y: y
            kwargs: kwargs
        """
        self.model.fit(X, y.squeeze())

    def predict(self, X, **kwargs) -> np.ndarray:
        """predict.

        Args:
            X: X
            kwargs: kwargs
        """
        return self.model.predict(X)


class SVM(SKLearnModel):
    """LogisticRegression."""

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs: kwargs
        """
        super(SVM, self).__init__(**kwargs)
        self.model = SVC()

    def fit(self, X, y, **kwargs):
        """fit.

        Args:
            X: X
            y: y
            kwargs: kwargs
        """

        self.model.fit(X, y.squeeze())

    def predict(self, X, **kwargs) -> np.ndarray:
        """predict.

        Args:
            X: X
            kwargs: kwargs
        """
        return self.model.predict_proba(X)[:, 1]


class RandomGuess(SKLearnModel):
    """RandomGuess."""

    def __init__(self, **kwargs):
        """__init__.

        Args:
            solver (str): solver
            kwargs: kwargs
        """
        super(RandomGuess, self).__init__(**kwargs)
        self.freq = 0.5

    def fit(self, X, y, **kwargs):
        """fit.

        Args:
            X: X
            y: y
            kwargs: kwargs
        """
        self.freq = np.sum(y == 1) / y.shape[0]

    def predict(self, X, **kwargs) -> np.ndarray:
        """predict.

        Args:
            X: X
            kwargs: kwargs
        """
        rand_guesses = np.random.choice(
            [0, 1], X.shape[0], p=[1 - self.freq, self.freq]
        )
        return rand_guesses


class ConstantGuess(SKLearnModel):
    """ConstantGuess."""

    def __init__(self, **kwargs):
        """__init__.

        Args:
            solver (str): solver
            kwargs: kwargs
        """
        super(ConstantGuess, self).__init__(**kwargs)
        self.guess = 1
        self.freq = 0.5

    def fit(self, X, y, **kwargs):
        """fit.

        Args:
            X: X
            y: y
            kwargs: kwargs
        """
        self.freq = np.sum(y == 1) / y.shape[0]
        if self.freq < 0.5:
            self.guess = 0
        else:
            self.guess = 1

    def predict(self, X, **kwargs) -> np.ndarray:
        """predict.

        Args:
            X: X
            kwargs: kwargs
        """
        rand_guesses = np.ones(X.shape[0]) * self.guess
        return rand_guesses


class GPClassification(SKLearnModel):
    """GPClassification."""

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs: kwargs
        """
        super(GPClassification, self).__init__(**kwargs)
        self.kernel = gaussian_process.kernels.RBF(1.0)
        self.model = gaussian_process.GaussianProcessClassifier(
            kernel=self.kernel,
        )

    def fit(self, X, y, **kwargs):
        """fit.

        Args:
            X: X
            y: y
            kwargs: kwargs
        """

        self.model.fit(X, y.squeeze())

    def predict(self, X, **kwargs) -> np.ndarray:
        """predict.

        Args:
            X: X
            kwargs: kwargs
        """
        return self.model.predict_proba(X)[:, 1]


class GPRegression(SKLearnModel):
    """GPRegression."""

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs: kwargs
        """
        super(GPRegression, self).__init__(**kwargs)
        self.kernel = gaussian_process.kernels.RBF(1.0)
        self.model = gaussian_process.GaussianProcessRegressor(
            kernel=self.kernel,
        )

    def fit(self, X, y, **kwargs):
        """fit.

        Args:
            X: X
            y: y
            kwargs: kwargs
        """

        self.model.fit(X, y.squeeze())

    def predict(self, X, **kwargs) -> np.ndarray:
        """predict.

        Args:
            X: X
            kwargs: kwargs
        """
        return self.model.predict(X)


class KNNClassifier(SKLearnModel):
    """KNNClassifier"""

    def __init__(self, n_neighbors: int, **kwargs):
        """__init__.

        Args:
            n_neighbors (int): Number of neighbors
            kwargs: kwargs
        """
        super(KNNClassifier, self).__init__(**kwargs)
        # Use precomputed matrix
        self.model = neighbors.KNeighborsClassifier(
            n_neighbors=n_neighbors, algorithm="brute", metric="precomputed"
        )
        self.use_model = True
        self.model_out = None

    def fit(self, X, y, **kwargs):
        """fit.

        For this model, make sure that X is a pairwise distance matrix in the
        fit loop

        Args:
            X: X
            y: y
            kwargs: kwargs
        """

        if len(np.unique(y)) == 1:
            self.use_model = False
            self.model_out = list(np.unique(y))[0]
        else:
            self.model.fit(X, y.squeeze())

    def predict(self, X, **kwargs) -> np.ndarray:
        """predict.

        Args:
            X: X
            kwargs: kwargs
        """
        # Note, this assumes that we have binary inputs trained
        # If we have 1 class, let's only return that single class...
        if self.use_model:

            return self.model.predict_proba(X)[:, 1]
        else:
            # Return the only class from the train set
            output = np.ones(len(X)).reshape(-1, 1)
            assert self.model_out is not None
            output = output * self.model_out
            return output


class KNNRegressor(SKLearnModel):
    """KNNRegressor."""

    def __init__(self, n_neighbors: int, **kwargs):
        """__init__.

        Args:
            n_neighbors (int): Number of neighbors
            kwargs: kwargs
        """
        super(KNNRegressor, self).__init__(**kwargs)
        # Use precomputed matrix
        self.model = neighbors.KNeighborsRegressor(
            n_neighbors=n_neighbors, algorithm="brute", metric="precomputed"
        )

    def fit(self, X, y, **kwargs):
        """fit.

        For this model, make sure that X is a pairwise distance matrix in the
        fit loop

        Args:
            X: X
            y: y
            kwargs: kwargs
        """

        self.model.fit(X, y.squeeze())

    def predict(self, X, **kwargs) -> np.ndarray:
        """predict.

        Args:
            X: X
            kwargs: kwargs
        """
        return self.model.predict(X)
