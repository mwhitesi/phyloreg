"""


"""
import h5py as h
import logging
import numpy as np

from itertools import product, izip
from functools import partial
from multiprocessing import Pool, cpu_count
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.graph import graph_laplacian
from warnings import warn

# C++ module
from _phyloreg import *

import matplotlib.pyplot as plt


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


def parallel_objective_by_example(O_i, w, species_graph_adjacency):
    p = 1.0 / (1.0 + np.exp(-np.dot(O_i, w)))
    return 2.0 * sum(species_graph_adjacency[k, l] * (p[k] - p[l])**2 for k in xrange(O_i.shape[0]) for l in xrange(k))


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
    def __init__(self, alpha=1.0, beta=0.0, normalize_laplacian=False, fit_intercept=False, opti_max_iter=1e4,
                 opti_lookahead_steps=50, opti_tol=1e-7, opti_learning_rate=1e-2, opti_learning_rate_decrease=1e-4,
                 random_seed=42, n_cpu=-1):
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
        self.opti_lookahead_steps = opti_lookahead_steps
        self.random_seed = random_seed
        self.o1list = []
        self.o2list = []
        self.o3list = []
        self.n_cpu = cpu_count() if n_cpu == -1 else n_cpu

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
        logging.debug("Preparing the data for the optimizer")

        # Push the species adjacency matrix to the C++ module's memory
        set_species_adjacency(species_graph_adjacency)

        # If required, add a feature for each example that serves as bias
        if self.fit_intercept:
            X = np.hstack((X, np.ones(X.shape[0]).reshape(-1, 1)))

        # Push the examples and labels to the C++ module's memory
        set_examples_and_labels(X, np.asarray(y, dtype=np.double))

        # Precompute the example ortholog feature matrices
        idx_by_species = dict(zip(species_graph_names, range(len(species_graph_names))))
        ortholog_matrix_by_example = []
        for i, x_i in enumerate(X):
            # H5py doesn't support integer keys
            if isinstance(orthologs, h.File):
                i = str(i)

            # Load the orthologs of X and create a matrix that also contains x
            x_orthologs_species = [idx_by_species[s] for s in orthologs[i]["species"]]
            x_orthologs_feats = orthologs[i]["X"]
            if self.fit_intercept:
                x_orthologs_feats = np.hstack((x_orthologs_feats, np.ones(x_orthologs_feats.shape[0]).reshape(-1, 1)))  # Add this bias term

            O_i = np.zeros((len(species_graph_names), len(x_i)))
            O_i[x_orthologs_species] = x_orthologs_feats
            O_i[idx_by_species[X_species[i]]] = x_i

            # Push the orthologs to the C++ module's memory
            set_example_orthologs(i, O_i)


        logging.debug("Initiating the optimization procedure (gradient ascent)")

        # Initialize parameters
        random_generator = np.random.RandomState(self.random_seed)
        w = np.zeros( X.shape[1]) #random_generator.rand(X.shape[1])

        # Initialize lookahead
        logging.debug("Computing the objective function")
        objective_t1, objective_t2, objective_t3 = get_objective(w)
        last_objective_checkpoint = objective_t1 - self.alpha * objective_t2 - self.beta * objective_t3

        # Ascend that gradient!
        iterations = 0
        shuffled_example_idx = np.arange(X.shape[0])
        random_generator.shuffle(shuffled_example_idx)
        while iterations < self.opti_max_iter:
            # The example considered by SGD
            iteration_example_idx = shuffled_example_idx[iterations % X.shape[0]]

            learning_rate = self.opti_learning_rate / (1.0 + self.opti_learning_rate_decrease * iterations)
            logging.debug("Iteration %d -- Objective: %.6f -- Learning rate: %.6f -- Example idx: %d" % (iterations, last_objective_checkpoint, learning_rate, iteration_example_idx))

            logging.debug("Computing the gradient")
            gradient_t1, gradient_t2, gradient_t3 = get_gradient(w, iteration_example_idx)
            gradient = gradient_t1 - self.alpha * gradient_t2 - self.beta * gradient_t3

            update = learning_rate * gradient
            w += update
            iterations += 1

            # Verify progress after a certain number of iterations
            if iterations % self.opti_lookahead_steps == 0:
                logging.debug("Computing the objective function")
                objective_t1, objective_t2, objective_t3 = get_objective(w)
                objective_val = objective_t1 - self.alpha * objective_t2 - self.beta * objective_t3

                # If there has been no change since the last objective check, stop.
                if np.abs(objective_val - last_objective_checkpoint) <= self.opti_tol:
                    logging.debug("Converged in %d iterations" % iterations)
                    break
                else:
                    # Otherwise, update the current objective value
                    last_objective_checkpoint = objective_val

        else:  # Executed if the loop ends after its condition is violated (not on break)
            logging.debug("The maximum number of iterations was reached prior to convergence. Try increasing the number"
			    " of iterations. opti_max_iter: %s", self.opti_max_iter)
            warn("The maximum number of iterations was reached prior to convergence. Try increasing the number of "
                 "iterations.")

        self.w = w
        reset()

        # plot objective values

        # print 'o1: ', self.o1list
        # print 'o2: ', self.o2list
        # print 'o3: ', self.o3list
        #
        # fig = plt.figure()
        #
        # ax1 = fig.add_subplot(311)
        # ax1.set_ylim([ min(self.o1list), max( self.o1list)])
        # ax1.plot( range(iterations + 1 ), np.asarray( self.o1list, dtype= float), 'r-')
        #
        # ax2 = fig.add_subplot(312)
        # ax2.set_ylim([ min(self.o2list), max( self.o2list)])
        # ax2.plot( range(iterations + 1), np.asarray( self.o2list, dtype= float), 'r-')
        #
        # ax3 = fig.add_subplot(313)
        # ax3.set_ylim([ min(self.o3list), max( self.o3list)])
        # ax3.plot( range(iterations + 1), np.asarray( self.o3list, dtype= float), 'r-')
        #
        # plt.savefig( 'alpha' + str( self.alpha ) + 'beta' + str( self.beta) + '.pdf' )



        # # Newton-Raphson method
        # # TODO: Can we use iterative reweighted least squares? I'm pretty sure we can.
        # # TODO: Check if our method can also be expressed as a weighted least squares (see hastie: p. 121)
        # # TODO: might be a bit faster.
        #
        # w = np.zeros(X.shape[1])  # Hastie says zero is a good starting point
        #
        #
        # iterations = 0
        # while iterations < self.opti_max_iter:
        #     objective_val = objective(w)
        #
        #     logging.debug(' Objective Value: %s                   Iterations: %s', objective_val, iterations )
        #     print( 'Iterations: ', iterations )
        #
        #     p = 1.0 / (1.0 + np.exp(-np.dot(X, w)))
        #     logging.debug('p: %s', p )
        #     #logging.debug(' w: %s', w )
        #
        #     #gradient = np.dot(X.T, y - p) - 2 * self.alpha * w - 4 * self.beta * np.dot(OLO, w)
        #     gradient = np.dot(X.T, y - p) / X.shape[0] - 2.0 * self.alpha * w - (4.0 * self.beta * np.dot(OLO, w) ) / (X.shape[0] * L.shape[0]**2)
        #
        #
        #
        #     W = np.diag(p * (1.0 - p))
        #     subgradient = -np.dot(np.dot(X.T, W), X) / X.shape[0] - 2 * self.alpha * np.eye(X.shape[1]) - (4 * self.beta * OLO ) / (X.shape[0] * L.shape[0]**2)
        #
        #     update = np.dot(np.linalg.inv(subgradient), gradient)
        #     w -= update
        #     iterations += 1
        #
        #     # Check for convergence
        #     if np.linalg.norm(update, ord=2) <= self.opti_tol:
        #         break
        #
        # self.w = w

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

    def _check_sgd(self):
        reset()

        print "Gradient verification for the Stochastic Gradient Descent"
        n_species = 5
        n_examples = 10
        n_features = 5
        epsilon = 1e-6
        alpha = np.random.rand() * 100
        beta = np.random.rand() * 100

        # Initialization
        X = np.random.rand(n_examples, n_features)
        y = np.random.randint(0, 2, n_examples)
        set_examples_and_labels(X, np.asarray(y, dtype=np.double))

        w = np.random.rand(n_features)

        # adjacency matrix

        A = np.random.rand(n_species, n_species)
        A = np.dot(A.T, A)  # Make it symmetric
        np.fill_diagonal(A, 100.)
        set_species_adjacency(A)

        # # Generate ortholog data
        for i, x_i in enumerate(X):
            O_i = np.random.rand(n_species, len(x_i))
            O_i[0] = x_i
            # Push the orthologs to the C++ module's memory
            set_example_orthologs(i, O_i)

        g1 = np.zeros(n_features, dtype=float)
        g2 = np.zeros(n_features, dtype=float)
        g3 = np.zeros(n_features, dtype=float)
        # g4 = np.zeros(n_examples, dtype=float)

        for iteration_example_idx in range(n_examples):
            # Compute the gradient according to the SGD
            gradient_t1, gradient_t2, gradient_t3 = get_gradient(w, iteration_example_idx)
            g1 += gradient_t1
            g2 += alpha * gradient_t2
            g3 += beta * gradient_t3

        g1 /= n_examples
        g2 /= n_examples
        g3 /= n_examples

        g4 = g1 - g2 - g3

        # Compute the empirical gradient estimate

        # Check the gradient for each component of w
        for i in xrange(w.shape[0]):
            w_1 = w.copy()
            w_2 = w.copy()
            w_1[i] += epsilon
            w_2[i] -= epsilon
            objective_t11, objective_t21, objective_t31 = get_objective(w_1)
            objective_t12, objective_t22, objective_t32 = get_objective(w_2)
            objective_val1 = objective_t11 - alpha * objective_t21 - beta * objective_t31
            objective_val2 = objective_t12 - alpha * objective_t22 - beta * objective_t32
            empirical_g1 = (objective_t11 - objective_t12) / (2 * epsilon)
            empirical_g2 = (objective_t21 - objective_t22) / (2 * epsilon)
            empirical_g3 = (objective_t31 - objective_t32) / (2 * epsilon)
            empirical_g4 = (objective_val1 - objective_val2) / (2 * epsilon)

            if not np.allclose(empirical_g1, g1[i]):
                print "FAILED. Expected gradient: %.8f   Calculated gradient: %.8f" % (empirical_g1, g1[i])
            if not np.allclose(empirical_g2, g2[i]):
                print "FAILED. Expected gradient: %.8f   Calculated gradient: %.8f" % (empirical_g2, g2[i])
            if not np.allclose(empirical_g3, g3[i]):
                print "FAILED. Expected gradient: %.8f   Calculated gradient: %.8f" % (empirical_g3, g3[i])
            if not np.allclose(empirical_g4, g4[i]):
                print "FAILED. Expected gradient: %.8f   Calculated gradient: %.8f" % (empirical_g4, g4[i])
        else:
            print "PASSED. "
            #return True

        reset()



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
