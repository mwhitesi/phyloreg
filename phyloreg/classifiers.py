"""


"""
import h5py as h
import numpy as np
import sys
from treestructure import getTreeDict

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.graph import graph_laplacian


# TODO: Include a intercept term
class RidgeRegression(BaseEstimator, ClassifierMixin):
    """Ridge regression species-level with phylogenetic regularization

    Parameters
    ----------
    alpha : float
        Hyperparameter for the L2 norm regularizer. Greater values lead to stronger regularization.
    beta : float
        Hyperparameter for the phylogenetic regularization. Greater values lead to stronger regularization. Zero discards
        the phylogenetic regularizer and leads to regular ridge regression.
    normalize_laplacian: bool
        Whether or not to normalize the graph laplacian in the phylogenetic regularizer.
    fit_intercept: bool
        Whether to calculate the intercept for this model. If set to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    Attributes
    ----------
    w : array_like, dtype=float
        The fitted model's coefficients.
    intercept: float
        The intercept of the model.

    """
    def __init__(self, alpha=1.0, beta=1.0, normalize_laplacian=False, fit_intercept=False):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.normalize_laplacian = normalize_laplacian
        self.fit_intercept = fit_intercept
        self.w = None
        self.intercept = 0

    def fit(self, X, X_species, y, orthologs, species_adjacency):
        """Fit the model

        Parameters
        ----------
        X: array_like, dtype=float, shape=(n_examples, n_features)
            The feature vectors of each labeled example.
        X_species: array_like, dtype=float, shape=(n_examples,)
            The species to which each example belongs.
        y: array_like, dtype=float, shape(n_examples,)
            The labels of the examples in X.
        orthologs: dict
            A dictionnary in which the keys are indices of X and the values are another dict, which contain
            the orthologous sequences and their species. TIP: use an HDF5 file to store this information if the data
            doesn't fit into memory. Note: assumes that there is at most 1 ortholog per species.

            ex: {0: {"species": [0, 3, 5],
                     "X": [[0, 2, 1, 4],    # Ortholog 1
                           [9, 4, 3, 1],    # Ortholog 2
                           [0, 0, 2, 1]]},  # Ortholog 3
                 1: {"species": [1, 3],
                     "X": [[1, 4, 7, 6],
                           [4, 4, 9, 3]]}}
        species_adjacency: array_like, dtype=float, shape=(n_species, n_species)
            The adjacency matrix of the species graph. The species indices must match between X_species and orthologs.

        Note
        ----
        It is recommended to center the features vectors for the examples and their orthologs using a standard scaler.
        (see: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

        """
        # Find out how many species to consider
        n_species = species_adjacency.shape[0]

        if self.fit_intercept:
            raise NotImplementedError("Not done yet!")
            #y_centered = (y - np.mean(y))/np.std(y)
        else:
            y_centered = y

        # Precompute the laplacian of the species graph
        # Note: we assume that there is one entry per species. This sacrifices a bit of memory, but allows the precomputation
        #       the graph laplacian.
        L = graph_laplacian(species_adjacency, normed=self.normalize_laplacian)
        L *= 2 * self.beta

        matrix_to_invert = np.zeros((X.shape[1], X.shape[1]))

        # Compute the Phi^t x L x Phi product, where L is the block diagonal matrix with blocks equal to variable L
        for i, x in enumerate(X):
            # H5py doesn't support integer keys
            if isinstance(orthologs, h.File):
                i = str(i)
            # Load the orthologs of X and create a matrix that also contains x
            x_orthologs_species = orthologs[i]["species"]
            x_orthologs_feats = orthologs[i]["X"]

            X_tmp = np.zeros((n_species, x.shape[0]))
            X_tmp[x_orthologs_species] = x_orthologs_feats
            X_tmp[X_species[i]] = x

            # Compute the efficient product and add it to the nasty product
            matrix_to_invert += np.dot(np.dot(X_tmp.T, L), X_tmp)

        # Compute the Phi^T x Phi matrix product that includes the labeled examples only
        matrix_to_invert += np.dot(X.T, X)

        # Compute the alpha * I product
        matrix_to_invert += self.alpha * np.eye(X.shape[1])

        # Compute the value of w, the predictor that minimizes the objective function
        self.w = np.dot(np.dot(np.linalg.inv(matrix_to_invert), X.T), y_centered).reshape(-1,)

        if self.fit_intercept:
            self.intercept = 0
            raise NotImplementedError()
        else:
            self.intercept = 0

    def predict(self, X):
        """Compute predictions using the learned model

        Parameters
        ----------
        X: array_like, dtype=float, shape=(n_examples, n_features)
            The feature vectors of the examples for which a prediction is required.

        Returns
        -------
        predictions: array_like, dtype=float, shape=(n_examples,)
            The predictions computed using the model.

        Note
        ----
        If scaling was used for fitting, the same transformation must be applied to X before computing predictions.

        """
        if self.w is None:
            raise RuntimeError("The algorithm must be fitted first!")
        return np.dot(X, self.w).reshape(-1,)


if __name__ == "__main__":


    # 100 Way alignment tree file name with full path
    treefilename = 'modtree100'

    # create tree dict
    phylo = getTreeDict( treefilename )
    print ( phylo )
    sys.exit()

    # Check if when beta=0 we recover regular ridge regression

    X = np.random.rand(10, 5)
    y = np.random.rand(10)

    A = np.random.rand(3, 3)
    A = np.dot(A.T, A)

    should_be_ridge = RidgeRegression(alpha=1.0, beta=0.0)
    should_be_ridge.fit(X=X,
                        X_species=[0] * len(y),
                        y=y,
                        orthologs=dict([(i, {"species": [0, 1, 2], "X": np.random.rand(3, 5)}) for i in xrange(X.shape[0])]),
                        species_adjacency=A)

    regular_ridge = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + np.eye(X.shape[1])), X.T), y)

    assert np.allclose(should_be_ridge.w, regular_ridge)
