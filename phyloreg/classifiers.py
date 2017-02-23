"""


"""
import h5py as h
import logging
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.graph import graph_laplacian
from warnings import warn


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
        The fitted model's coefficients (last value is the intercept if fit_intercept=True).

    """
    def __init__(self, alpha=1.0, beta=1.0, normalize_laplacian=False, fit_intercept=False):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.normalize_laplacian = normalize_laplacian
        self.fit_intercept = fit_intercept
        self.w = None

    def fit(self, X, X_species, y, orthologs, species_graph_adjacency, species_graph_names):
        """Fit the model

        Parameters
        ----------
        X: array_like, dtype=float, shape=(n_examples, n_features)
            The feature vectors of each labeled example.
        X_species: array_like, dtype=str, shape=(n_examples,)
            The name of the species to which each example belongs.
        y: array_like, dtype=float, shape(n_examples,)
            The labels of the examples in X.
        orthologs: dict
            A dictionnary in which the keys are indices of X and the values are another dict, which contain
            the orthologous sequences and their species names. TIP: use an HDF5 file to store this information if the data
            doesn't fit into memory. Note: assumes that there is at most 1 ortholog per species.

            ex: {0: {"species": ["species1", "species5", "species2"],
                     "X": [[0, 2, 1, 4],    # Ortholog 1
                           [9, 4, 3, 1],    # Ortholog 2
                           [0, 0, 2, 1]]},  # Ortholog 3
                 1: {"species": ["species1", "species3"],
                     "X": [[1, 4, 7, 6],
                           [4, 4, 9, 3]]}}
        species_graph_adjacency: array_like, dtype=float, shape=(n_species, n_species)
            The adjacency matrix of the species graph.
        species_graph_names: array_like, dtype=str, shape(n_species,)
            The names of the species in the graph. The names should follow the same order as the adjacency matrix.
            ex: If species_graph_names[4] relates to species_graph_adjacency[4] and species_graph_adjacency[:, 4].

        Note
        ----
        It is recommended to center the features vectors for the examples and their orthologs using a standard scaler.
        (see: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

        """
        # Create a mapping between species names and indices in the graph adjacency matrix
        idx_by_species = dict(zip(species_graph_names, range(len(species_graph_names))))

        if self.fit_intercept:
            X = np.hstack((X, np.ones(X.shape[0]).reshape(-1, 1)))  # Add a feature for each example that serves as bias

        # Precompute the laplacian of the species graph
        # Note: we assume that there is one entry per species. This sacrifices a bit of memory, but allows the precomputation
        #       the graph laplacian.
        L = graph_laplacian(species_graph_adjacency, normed=self.normalize_laplacian)
        L *= 2.0 * self.beta

        matrix_to_invert = np.zeros((X.shape[1], X.shape[1]))

        # Compute the Phi^t x L x Phi product, where L is the block diagonal matrix with blocks equal to variable L
        for i, x in enumerate(X):
            # H5py doesn't support integer keys
            if isinstance(orthologs, h.File):
                i = str(i)

            if len(orthologs[i]["species"]) > 0:
                # Load the orthologs of X and create a matrix that also contains x
                x_orthologs_species = [idx_by_species[s] for s in orthologs[i]["species"]]
                x_orthologs_feats = orthologs[i]["X"]
                if self.fit_intercept:
                    x_orthologs_feats = np.hstack((x_orthologs_feats, np.ones(x_orthologs_feats.shape[0]).reshape(-1, 1)))  # Add this bias term

                X_tmp = np.zeros((len(species_graph_names), x_orthologs_feats.shape[1]))
                X_tmp[x_orthologs_species] = x_orthologs_feats
                X_tmp[idx_by_species[X_species[i]]] = x

                # Compute the efficient product and add it to the nasty product
                matrix_to_invert += np.dot(np.dot(X_tmp.T, L), X_tmp)

        # Compute the Phi^T x Phi matrix product that includes the labeled examples only
        matrix_to_invert += np.dot(X.T, X)

        # Compute the alpha * I product
        matrix_to_invert += self.alpha * np.eye(X.shape[1])

        # Compute the value of w, the predictor that minimizes the objective function
        self.w = np.dot(np.dot(np.linalg.inv(matrix_to_invert), X.T), y).reshape(-1,)


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
        if self.fit_intercept:
            X = np.hstack((X, np.ones(X.shape[0]).reshape(-1, 1)))  # Add a feature for each example that serves as bias
        if self.w is None:
            raise RuntimeError("The algorithm must be fitted first!")
        return np.dot(X, self.w).reshape(-1,)


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """Logistic regression species-level with phylogenetic regularization

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
    opti_max_iter: int
        The maximum number of iterations for the optimization algorithm.
    opti_tol: float
        The minimum difference between the solution of two iterations to consider that convergence has been achieved
        by the optimization algorithm.

    Attributes
    ----------
    w : array_like, dtype=float
        The fitted model's coefficients (last value is the intercept if fit_intercept=True).

    """
    def __init__(self, alpha=1.0, beta=0.0, normalize_laplacian=False, fit_intercept=False, opti_max_iter=1e5,
                 opti_tol=1e-7, opti_learning_rate=1e-2, opti_learning_rate_decrease=1e-4, random_seed=42):
        # Classifier parameters
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.normalize_laplacian = normalize_laplacian
        self.fit_intercept = fit_intercept
        self.w = None
        self.intercept = 0

        # Optimisation algorithm parameters
        self.opti_tol = opti_tol
        self.opti_max_iter = opti_max_iter
        self.opti_learning_rate = opti_learning_rate
        self.opti_learning_rate_decrease = opti_learning_rate_decrease
        self.opti_lookahead_steps = 50
        self.random_seed = random_seed

    def fit(self, X, X_species, y, orthologs, species_graph_adjacency, species_graph_names):
        """Fit the model

        Parameters
        ----------
        X: array_like, dtype=float, shape=(n_examples, n_features)
            The feature vectors of each labeled example.
        X_species: array_like, dtype=str, shape=(n_examples,)
            The name of the species to which each example belongs.
        y: array_like, dtype=float, shape(n_examples,)
            The labels of the examples in X.
        orthologs: dict
            A dictionnary in which the keys are indices of X and the values are another dict, which contain
            the orthologous sequences and their species names. TIP: use an HDF5 file to store this information if the data
            doesn't fit into memory. Note: assumes that there is at most 1 ortholog per species.

            ex: {0: {"species": ["species1", "species5", "species2"],
                     "X": [[0, 2, 1, 4],    # Ortholog 1
                           [9, 4, 3, 1],    # Ortholog 2
                           [0, 0, 2, 1]]},  # Ortholog 3
                 1: {"species": ["species1", "species3"],
                     "X": [[1, 4, 7, 6],
                           [4, 4, 9, 3]]}}
        species_graph_adjacency: array_like, dtype=float, shape=(n_species, n_species)
            The adjacency matrix of the species graph.
        species_graph_names: array_like, dtype=str, shape(n_species,)
            The names of the species in the graph. The names should follow the same order as the adjacency matrix.
            ex: If species_graph_names[4] relates to species_graph_adjacency[4] and species_graph_adjacency[:, 4].

        Note
        ----
        It is recommended to center the features vectors for the examples and their orthologs using a standard scaler.
        (see: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

        """
        def objective(w):
            """
            The function that we want to maximize
            """
            o = 0.0

            # Likelihood
            for i in xrange(X.shape[0]):
                pi = 1.0 / (1.0 + np.exp(-np.dot(w, X[i])))
                o += np.log(pi) if y[i] == 1 else np.log(1.0 - pi)
            o /= X.shape[0]

            # L2 norm
            o -= self.alpha * np.linalg.norm(w, ord=2)**2

            # Manifold (computed for each example x_i)
            for i, x_i in enumerate(X):
                # H5py doesn't support integer keys
                if isinstance(orthologs, h.File):
                    i = str(i)

                if len(orthologs[i]["species"]) > 0:
                    # Load the orthologs of X and create a matrix that also contains x
                    x_orthologs_species = [idx_by_species[s] for s in orthologs[i]["species"]]
                    x_orthologs_feats = orthologs[i]["X"]
                    if self.fit_intercept:
                        x_orthologs_feats = np.hstack((x_orthologs_feats, np.ones(x_orthologs_feats.shape[0]).reshape(-1, 1)))  # Add this bias term

                    O_i = np.zeros((len(species_graph_names), x_orthologs_feats.shape[1]))
                    O_i[x_orthologs_species] = x_orthologs_feats
                    O_i[idx_by_species[X_species[i]]] = x_i

                    f = np.dot(O_i, w)
                    o -= self.beta * 2.0 * np.dot(np.dot(f.T, L), f) / (X.shape[0] * L.shape[0]**2)
            return o

        # Create a mapping between species names and indices in the graph adjacency matrix
        idx_by_species = dict(zip(species_graph_names, range(len(species_graph_names))))

        # If required, add a feature for each example that serves as bias
        if self.fit_intercept:
            X = np.hstack((X, np.ones(X.shape[0]).reshape(-1, 1)))

        # Compute the O^t x L x O product, where L is the block diagonal graph laplacian matrix
        L = graph_laplacian(species_graph_adjacency, normed=self.normalize_laplacian)
        OLO = np.zeros((X.shape[1], X.shape[1]))
        for i, x_i in enumerate(X):
            # H5py doesn't support integer keys
            if isinstance(orthologs, h.File):
                i = str(i)

            if len(orthologs[i]["species"]) > 0:
                # Load the orthologs of X and create a matrix that also contains x
                x_orthologs_species = [idx_by_species[s] for s in orthologs[i]["species"]]
                x_orthologs_feats = orthologs[i]["X"]
                if self.fit_intercept:
                    x_orthologs_feats = np.hstack((x_orthologs_feats, np.ones(x_orthologs_feats.shape[0]).reshape(-1, 1)))  # Add this bias term

                O_i = np.zeros((len(species_graph_names), x_orthologs_feats.shape[1]))
                O_i[x_orthologs_species] = x_orthologs_feats
                O_i[idx_by_species[X_species[i]]] = x_i

                # Compute the efficient product and add it to the nasty product
                OLO += np.dot(np.dot(O_i.T, L), O_i)

        logging.debug("Initiating gradient ascent")

        # Initialize parameters
        random_generator = np.random.RandomState(self.random_seed)
        w = random_generator.rand(X.shape[1])

        # Initialize lookahead
        lookahead_optimum = objective(w)
        lookahead_w = w
        lookahead_count = 0

        # Ascend that gradient!
        iterations = 0
        objective_val = lookahead_optimum
        while iterations < self.opti_max_iter:
            learning_rate = self.opti_learning_rate / (1.0 + self.opti_learning_rate_decrease * iterations)
            logging.debug("Iteration %d -- Objective: %.6f -- Learning rate: %.6f" % (iterations, objective_val, learning_rate))

            p = 1.0 / (1.0 + np.exp(-np.dot(X, w)))
            gradient = np.dot(X.T, y - p) / X.shape[0] - 2.0 * self.alpha * w - 4.0 * self.beta * np.dot(OLO, w) / (X.shape[0] * L.shape[0]**2)

            update = learning_rate * gradient
            w += update
            iterations += 1

            # Verify progress after each iteration
            objective_val = objective(w)

            # If we find a solution that is very close to the optimum register a "no change"
            if np.abs(objective_val - lookahead_optimum) <= self.opti_tol:
                lookahead_count += 1
            else:
                # Otherwise, update the current value
                lookahead_optimum = objective_val
                lookahead_w = w.copy()

            # Check for convergence (objective value has not significantly changed during lookahead)
            if lookahead_count >= self.opti_lookahead_steps:
                logging.debug("Converged in %d iterations" % (iterations - lookahead_count))
                w = lookahead_w
                break
        else:  # Executed if the loop ends after its condition is violated (not on break)
            logging.debug("The maximum number of iterations was reached prior to convergence. Try increasing the number"
                          " of iterations.")
            warn("The maximum number of iterations was reached prior to convergence. Try increasing the number of "
                 "iterations.")

        self.w = w

        # # Newton-Raphson method
        # # TODO: Can we use iterative reweighted least squares? I'm pretty sure we can.
        # # TODO: Check if our method can also be expressed as a weighted least squares (see hastie: p. 121)
        # # TODO: might be a bit faster.
        # w = np.zeros(X.shape[1])  # Hastie says zero is a good starting point
        # iterations = 0
        # while iterations < self.opti_max_iter:
        #     p = 1.0 / (1.0 + np.exp(-np.dot(X, w)))
        #     gradient = np.dot(X.T, y - p) - 2 * self.alpha * w - 4 * self.beta * np.dot(OLO, w)
        #
        #     W = np.diag(p * (1.0 - p))
        #     subgradient = -np.dot(np.dot(X.T, W), X) - 2 * self.alpha * np.eye(X.shape[1]) - 4 * self.beta * OLO
        #
        #     update = np.dot(np.linalg.inv(subgradient), gradient)
        #     w -= update
        #     iterations += 1
        #
        #     # Check for convergence
        #     if np.linalg.norm(update, ord=2) <= self.opti_tol:
        #         break

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
        if self.fit_intercept:
            X = np.hstack((X, np.ones(X.shape[0]).reshape(-1, 1)))  # Add a feature for each example that serves as bias
        if self.w is None:
            raise RuntimeError("The algorithm must be fitted first!")
        return 1.0 / (1.0 + np.exp(-np.dot(X, self.w).reshape(-1,)))

    def _check_likelihood_gradient(self):
        print "Gradient verification for the log likelihood term...",
        n_examples = 10
        n_features = 5
        epsilon = 1e-6

        # Initialization
        X = np.random.randint(0, 2, (n_examples, n_features))
        y = np.random.randint(0, 2, n_examples)
        w = np.random.rand(n_features)

        # Compute the gradient according to the expression
        gradient = np.dot(X.T, y - 1.0 / (1.0 + np.exp(-np.dot(X, w)))) / X.shape[0]

        # Compute the empirical gradient estimate
        def loss(w):
            l = 0.0
            for i in xrange(X.shape[0]):
                pi = 1.0 / (1.0 + np.exp(-np.dot(w, X[i])))
                l += np.log(pi) if y[i] == 1 else np.log(1.0 - pi)
            l /= X.shape[0]
            return l

        # Check the gradient for each component of w
        for i in xrange(w.shape[0]):
            w_1 = w.copy()
            w_2 = w.copy()
            w_1[i] += epsilon
            w_2[i] -= epsilon
            empirical_gradient = (loss(w_1) - loss(w_2)) / (2 * epsilon)
            if not np.allclose(empirical_gradient, gradient[i]):
                print "FAILED. Expected gradient: %.8f   Calculated gradient: %.8f" % (empirical_gradient, gradient[i])
                return False
        else:
            print "PASSED"
            return True

    def _check_l2_norm_gradient(self):
        print "Gradient verification for the L2 norm term...",
        n_features = 5
        alpha = np.random.rand() * 100.0
        epsilon = 1e-6

        # Initialization
        w = np.random.rand(n_features)

        # Compute the gradient according to the expression
        gradient = 2 * alpha * w

        # Compute the empirical gradient estimate
        def loss(w):
            return alpha * np.linalg.norm(w, ord=2)**2

        # Check the gradient for each component of w
        for i in xrange(w.shape[0]):
            w_1 = w.copy()
            w_2 = w.copy()
            w_1[i] += epsilon
            w_2[i] -= epsilon
            empirical_gradient = (loss(w_1) - loss(w_2)) / (2 * epsilon)
            if not np.allclose(empirical_gradient, gradient[i]):
                print "FAILED. Expected gradient: %.8f   Calculated gradient: %.8f" % (empirical_gradient, gradient[i])
                return False
        else:
            print "PASSED"
            return True

    def _check_manifold_gradient(self):
        print "Gradient verification for the manifold regularization term...",
        n_examples = 10
        n_features = 5
        beta = np.random.rand() * 100.0
        epsilon = 1e-6

        # Initialization
        O = np.random.randint(0, 2, (n_examples, n_features))
        A = np.random.rand(n_examples, n_examples)
        A = np.dot(A.T, A)  # Make it symmetric
        np.fill_diagonal(A, 100.)
        L = graph_laplacian(A)
        w = np.random.rand(n_features)

        # Compute the gradient according to the expression
        OLO = np.dot(np.dot(O.T, L), O)
        gradient = 4 * beta * np.dot(OLO, w) / (n_examples * L.shape[0]**2)

        # Compute the empirical gradient estimate
        def loss(w):
            l = 0.0
            for i in xrange(O.shape[0]):
                for j in xrange(O.shape[0]):
                    l += A[i, j] * (np.dot(w, O[i]) - np.dot(w, O[j]))**2
            l /= (n_examples * L.shape[0]**2)
            return beta * l

        # Check the gradient for each component of w
        for i in xrange(w.shape[0]):
            w_1 = w.copy()
            w_2 = w.copy()
            w_1[i] += epsilon
            w_2[i] -= epsilon
            empirical_gradient = (loss(w_1) - loss(w_2)) / (2 * epsilon)
            if not np.allclose(empirical_gradient, gradient[i]):
                print "FAILED. Expected gradient: %.8f   Calculated gradient: %.8f" % (empirical_gradient, gradient[i])
                return False
        else:
            print "PASSED"
            return True

    def _check_gradients(self):
        print "Gradient validation tests\n--------------------------------"
        if self._check_likelihood_gradient() and \
            self._check_l2_norm_gradient() and \
            self._check_manifold_gradient():
            print "---> All gradients are correct."


if __name__ == "__main__":
    LogisticRegression()._check_gradients(); exit()

    # Check if when beta=0 we recover regular ridge regression

    X = np.random.rand(10, 5)
    y = np.random.rand(10)

    A = np.random.rand(3, 3)
    A = np.dot(A.T, A)

    clf = LogisticRegression(alpha=1.0, beta=1.0)
    clf.fit(X=X,
                        X_species=[0] * len(y),
                        y=y,
                        orthologs=dict([(i, {"species": [0, 1, 2], "X": np.random.rand(3, 5)}) for i in xrange(X.shape[0])]),
                        species_graph_adjacency=A,
                        species_graph_names=[0, 1, 2])
    exit()

    should_be_ridge = RidgeRegression(alpha=1.0, beta=0.0)
    should_be_ridge.fit(X=X,
                        X_species=[0] * len(y),
                        y=y,
                        orthologs=dict([(i, {"species": [0, 1, 2], "X": np.random.rand(3, 5)}) for i in xrange(X.shape[0])]),
                        species_adjacency=A)

    regular_ridge = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + np.eye(X.shape[1])), X.T), y)

    assert np.allclose(should_be_ridge.w, regular_ridge)