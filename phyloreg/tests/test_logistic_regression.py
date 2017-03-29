from __future__ import print_function

import numpy as np
import sys

from unittest import TestCase

from .._phyloreg import *


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, ** kwargs)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class LogisticRegressionTests(TestCase):
    """
    Logistic regression tests
    """
    def setUp(self):
        """
        Called before each test

        """
        reset()

        self.n_species = 5
        self.n_examples = 100
        self.n_features = 100
        self.epsilon = 1e-5

        # Initialization
        self.X = np.random.rand(self.n_examples, self.n_features)
        self.y = np.random.randint(0, 2, self.n_examples)
        set_examples_and_labels(self.X, self.y.astype(np.double))

        self.w = np.random.rand(self.n_features) * np.random.randint(-1, 2, self.n_features)

        # adjacency matrix

        self.A = np.random.rand(self.n_species, self.n_species)
        self.A = np.dot(self.A.T, self.A)  # Make it symmetric
        np.fill_diagonal(self.A, self.A.max())
        np.testing.assert_almost_equal(self.A, self.A.T)
        set_species_adjacency(self.A)

        # # Generate ortholog data
        self.O = []
        for i, x_i in enumerate(self.X):
            O_i = np.random.rand(self.n_species, len(x_i))
            O_i[0] = x_i
            self.O.append(O_i)
            set_example_orthologs(i, O_i)

    def tearDown(self):
        """
        Called after each test

        """
        reset()

    def test_objective_likelihood(self):
        """
        Objective value: likelihood term

        """
        desired = 0.0
        p = sigmoid(np.dot(self.X, self.w))
        for i in xrange(self.n_examples):
            desired += self.y[i] * np.log(p[i]) + (1. - self.y[i]) * np.log(1. - p[i])
        desired /= self.n_examples
        np.testing.assert_almost_equal(actual=get_objective(self.w)[0], desired=desired)

    def test_objective_l2_norm(self):
        """
        Objective value: l2 norm term

        """
        np.testing.assert_almost_equal(actual=get_objective(self.w)[1], desired=np.linalg.norm(self.w, ord=2)**2)

    def test_objective_phylo(self):
        """
        Objective value: phylogenetic term

        """
        desired = 0.0
        for i in xrange(self.n_examples):
            p = sigmoid(np.dot(self.O[i], self.w))
            for k in xrange(self.n_species):
                for l in xrange(self.n_species):
                    desired += self.A[k, l] * (p[k] - p[l])**2
        desired /= self.n_examples * self.n_species * self.n_species
        np.testing.assert_almost_equal(actual=get_objective(self.w)[2], desired=desired)

    def _verify_gradient(self, term, name):
        # Compute the gradient (average of SGD gradients over all examples)
        g = np.zeros(self.n_features, dtype=np.double)
        for i in xrange(self.n_examples):
            # Compute the gradient according to the SGD
            g += get_gradient(self.w, i)[term]
        g /= self.n_examples

        # Check the gradient for each component of w
        for i in xrange(self.w.shape[0]):
            # Compute the empirical gradient estimate
            w_1 = self.w.copy()
            w_2 = self.w.copy()
            w_1[i] += self.epsilon
            w_2[i] -= self.epsilon
            empirical_g = (get_objective(w_1)[term] - get_objective(w_2)[term]) / (2 * self.epsilon)

            np.testing.assert_almost_equal(actual=g[i], desired=empirical_g,
                                           err_msg="%s term gradient is incorrect" % name.title(), decimal=7)

    def test_gradient_likelihood(self):
        """
        Gradients: empirical verification of the likelihood term gradient

        """
        self._verify_gradient(0, "likelihood")

    def test_gradient_l2_norm(self):
        """
        Gradients: empirical verification of the L2 norm term gradient

        """
        self._verify_gradient(1, "l2 norm")

    def test_gradient_phylo(self):
        """
        Gradients: empirical verification of the phylogenetic term gradient

        """
        self._verify_gradient(2, "phylogenetic")