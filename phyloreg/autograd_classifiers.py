"""
	Phyloreg: phylogenetic regularization
	Copyright (C) 2017 Alexandre Drouin, Faizy Ahsan
	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.
	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.
	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import logging

from autograd import grad
from autograd import numpy as np
from autograd.optimizers import sgd
from functools import partial
import h5py as h
from math import ceil
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_X_y
from sklearn.utils.graph import graph_laplacian


def sigmoid(x):
	return 1.0 / (1 + np.exp(x))


class NoImprovementException(Exception):
	"""
	An exception that is used to signal that the objective function has not
	improved for a certain number of iterations.

	"""
	def __init__(self, how_long=-1):
		self.how_long = how_long


class NanInfException(Exception):
	"""
	An exception that is used to signal that a nan of inf value was found in
	the gradient.

	"""
	pass


class MiniBatchOptimizer(object):
	"""
	Mini-bacth optimization by gradient descent

	"""
	def __init__(self, learning_rate, patience, tolerance, batch_size,
				 max_epochs, gradient_clip_norm=np.infty):
		self.batch_size = batch_size
		self.gradient_clip_norm = gradient_clip_norm
		self.learning_rate = learning_rate
		self.max_epochs = max_epochs
		self.patience = patience
		self.tolerance = tolerance

	def minimize(self, learner, X, y, X_orthologs, species_graph_adjacency):
		"""
		Assumes that the learner has the following methods:
		1. _objective: computes the objective function on the entire training set
		2. _objective_batch: computes the objective on a mini batch

		"""
		# Define the mini batch behavior
		def get_batch_idx(iter_nb, n_batches, batch_size):
			idx = iter_nb % n_batches
			return slice(idx * batch_size, (idx + 1) * batch_size)
		get_batch_idx = partial(get_batch_idx, n_batches=int(ceil(1.0 * X.shape[0] / self.batch_size)), batch_size=self.batch_size)

		# Find the number of batches per epoch
		epoch_steps = int(ceil(1.0 * X.shape[0] / self.batch_size))

		# Define objective function and compute its gradient
		batch_objective = partial(learner._objective_batch, X=X, y=y, X_orthologs=X_orthologs, species_graph_adjacency=species_graph_adjacency, get_batch_idx=get_batch_idx, n_examples=X.shape[0], n_features=X.shape[1])
		objective = partial(learner._objective, X=X, y=y, X_orthologs=X_orthologs, species_graph_adjacency=species_graph_adjacency, n_examples=X.shape[0], n_features=X.shape[1])
		batch_gradient = grad(batch_objective)

		def clipped_gradient(*argv):
			"""
			A wrapper that performs gradient clipping on the gradient function

			"""
			v = self.gradient_clip_norm
			g = batch_gradient(*argv)
			if np.linalg.norm(g) > v:
				g = (g / np.linalg.norm(g)) * v
			return g

		# Callback function for logging optimization progress
		best_obj = dict(val=np.infty, since=0, w=None)
		def log_progress(params, iter_nb, gradient):

			# Verify that we don't have incorrect values in the gradient
			if np.isnan(gradient).sum() > 0 or np.isinf(gradient).sum() > 0:
				print gradient
				raise NanInfException()

			# On epoch completion
			if iter_nb % epoch_steps == 0:
				# Compute the objective on the entire training set
				o = objective(params)
				if best_obj["val"] - o > self.tolerance:
					# Case: the objective function has improved
					best_obj["val"] = o
					best_obj["since"] = 0
					best_obj["w"] = params.copy()
				else:
					# Case: the objective function has not improved (equal or worst)
					best_obj["since"] += 1

				# Log progress information
				epoch_nb = int(iter_nb / epoch_steps)
				train_auc = roc_auc_score(y_true=y, y_score=learner._predict(X, params))
				logging.debug("Epoch {0:d} (it: {1:d}) -- Obj.: {2:.4f} -- Best obj.: {3:.4f} -- Train AUC: {4:.4f}".format(epoch_nb, iter_nb, o, best_obj["val"], train_auc))

				# Check for stopping conditions
				if best_obj["since"] >= self.patience:
					raise NoImprovementException(best_obj["since"])

		# Parameter initialization
		w = np.random.rand(X.shape[1])

		# Gradient decent
		try:
			sgd(clipped_gradient, w, callback=log_progress, num_iters=self.max_epochs * epoch_steps, step_size=self.learning_rate, mass=0.9)
			logging.debug("The optimizer reached the maximum number of epochs without converging. Consider increasing it.")
		except NoImprovementException as e:
			logging.debug("The objective has not improved for {0:d} iterations. Stopping.".format(e.how_long))
		except NanInfException:
			logging.debug("A nan or inf value was encountered in the gradient. Stopping.")

		# Return the parameters that match the best observed objective value
		return best_obj["w"]


def generate_ortholog_feature_matrices(X, X_species, orthologs, species_graph_names, fit_intercept):
	# Create a mapping between species names and indices in the graph adjacency matrix
	idx_by_species = dict(zip(species_graph_names, range(len(species_graph_names))))

	X_orthologs = []
	for i, x in enumerate(X):
		# H5py doesn't support integer keys
		if isinstance(orthologs, h.File):
			i = str(i)

		# Load the orthologs of X and create a matrix that also contains x
		X_tmp = np.zeros((len(species_graph_names), X.shape[1]))
		X_tmp[idx_by_species[X_species[i]]] = x

		if len(orthologs[i]["species"]) > 0:
			x_orthologs_species = [idx_by_species[s] for s in orthologs[i]["species"]]
			x_orthologs_feats = orthologs[i]["X"]
			if fit_intercept:
				x_orthologs_feats = np.hstack((x_orthologs_feats, np.ones(x_orthologs_feats.shape[0]).reshape(-1, 1)))  # Add this bias term
			X_tmp[x_orthologs_species] = x_orthologs_feats

		X_orthologs.append(X_tmp)

	return X_orthologs


class BasePhyloLearner(object):
	def __init__(self):
		self.w = None

	def _objective(self, w, X, y, X_orthologs, species_graph_adjacency,
				   n_examples, n_features):
		raise NotImplementedError()

	def _objective_batch(self, w, iter_nb, X, y, X_orthologs,
						 species_graph_adjacency, get_batch_idx,
						 n_examples, n_features):
		"""
		Compute the objective function on a mini-batch

		"""
		batch_slice = get_batch_idx(iter_nb)
		return self._objective(w, X[batch_slice], y[batch_slice],
							   X_orthologs[batch_slice], species_graph_adjacency,
							   n_examples, n_features)

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
		X, y = check_X_y(X, y)

		if self.fit_intercept:
			X = np.hstack((X, np.ones(X.shape[0]).reshape(-1, 1)))  # Add a feature for each example that serves as bias

		# Generate the ortholog feature matrix for each example
		X_orthologs = generate_ortholog_feature_matrices(X, X_species, orthologs, species_graph_names, self.fit_intercept)

		# Optimize the objective function
		optimizer = MiniBatchOptimizer(learning_rate=self.opti_lr,
									   patience=self.opti_patience,
									   tolerance=self.opti_tol,
									   batch_size=self.opti_batch_size,
									   max_epochs=self.opti_max_epochs,
									   gradient_clip_norm=self.opti_clip_norm)
		self.w = optimizer.minimize(self, X, y, X_orthologs, species_graph_adjacency)

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
		if self.fit_intercept:
			X = np.hstack((X, np.ones(X.shape[0]).reshape(-1, 1)))  # Add a feature for each example that serves as bias
		return self._predict(X)


class AutogradRidgeRegression(BaseEstimator, ClassifierMixin, BasePhyloLearner):
	"""Ridge regression species-level with phylogenetic regularization

	Autograd is used to compute the objective function gradients and the objective
	is minimized using gradient descent.

	Parameters
	----------
	alpha : float
		Hyperparameter for the L2 norm regularizer. Greater values lead to stronger regularization.
	beta : float
		Hyperparameter for the phylogenetic regularization. Greater values lead to stronger regularization. Zero discards
		the phylogenetic regularizer and leads to regular ridge regression.
	fit_intercept: bool
		Whether to calculate the intercept for this model. If set to false, no intercept will be used in calculations
		(e.g. data is expected to be already centered).

	Attributes
	----------
	w : array_like, dtype=float
		The fitted model's coefficients (last value is the intercept if fit_intercept=True).

	"""
	def __init__(self, alpha=1.0, beta=1.0, fit_intercept=False, opti_lr=1e-4,
				 opti_max_epochs=1e2, opti_tol=1e-6, opti_patience=5,
				 opti_batch_size=10, opti_clip_norm=10.):
		self.alpha = float(alpha)
		self.beta = float(beta)
		self.fit_intercept = fit_intercept
		self.opti_lr = opti_lr
		self.opti_max_epochs = opti_max_epochs
		self.opti_tol = opti_tol
		self.opti_batch_size = opti_batch_size
		self.opti_patience = opti_patience
		self.opti_clip_norm = float(opti_clip_norm)
		super(AutogradRidgeRegression, self).__init__()

	def _objective_squared_loss(self, w, X, y):
		return np.linalg.norm(np.dot(X, w) - y, ord=2)**2

	def _objective_l2_norm(self, w):
		return np.dot(w, w)

	def _objective_orthologs(self, w, X_orthologs, species_graph_adjacency):
		n_species = species_graph_adjacency.shape[0]
		loss = 0.0
		for orthologs in X_orthologs:
			# An efficient way to compute the manifold part of the loss
			ortholog_predictions = np.dot(orthologs, w)
			loss += np.sum(species_graph_adjacency * \
						   (np.tile(ortholog_predictions,
									n_species).reshape(n_species, -1).T - \
									ortholog_predictions)**2)
		return loss

	def _objective(self, w, X, y, X_orthologs, species_graph_adjacency,
				   n_examples, n_features):
		return self._objective_squared_loss(w, X, y) / n_examples + \
			   self.alpha * self._objective_l2_norm(w) + \
			   self.beta * self._objective_orthologs(w, X_orthologs, species_graph_adjacency) / n_examples

	def _predict(self, X, w=None):
		if w is None:
			w = self.w
		return np.dot(X, w).reshape(-1,)


class AutogradLogisticRegression(BaseEstimator, ClassifierMixin, BasePhyloLearner):
	"""Logistic regression species-level with phylogenetic regularization

	Autograd is used to compute the objective function gradients and the objective
	is minimized using gradient descent.

	Parameters
	----------
	alpha : float
		Hyperparameter for the L2 norm regularizer. Greater values lead to stronger regularization.
	beta : float
		Hyperparameter for the phylogenetic regularization. Greater values lead to stronger regularization. Zero discards
		the phylogenetic regularizer and leads to regular ridge regression.
	fit_intercept: bool
		Whether to calculate the intercept for this model. If set to false, no intercept will be used in calculations
		(e.g. data is expected to be already centered).

	Attributes
	----------
	w : array_like, dtype=float
		The fitted model's coefficients (last value is the intercept if fit_intercept=True).

	"""
	def __init__(self, alpha=1.0, beta=1.0, fit_intercept=False, opti_lr=1e-4,
				 opti_max_epochs=1e2, opti_tol=1e-6, opti_patience=5,
				 opti_batch_size=10, opti_clip_norm=10.):
		self.alpha = float(alpha)
		self.beta = float(beta)
		self.fit_intercept = fit_intercept
		self.opti_lr = opti_lr
		self.opti_max_epochs = opti_max_epochs
		self.opti_tol = opti_tol
		self.opti_batch_size = opti_batch_size
		self.opti_patience = opti_patience
		self.opti_clip_norm = float(opti_clip_norm)
		super(AutogradLogisticRegression, self).__init__()

	def _objective_negative_log_likelihood(self, w, X, y):
		x_predictions = sigmoid(np.dot(X, w))
		y_probabilities = x_predictions * y + (1. - x_predictions) * (1. - y)
		return -np.sum(np.log(y_probabilities))

	def _objective_l2_norm(self, w):
		return np.dot(w, w)

	def _objective_orthologs(self, w, X_orthologs, species_graph_adjacency):
		n_species = species_graph_adjacency.shape[0]
		loss = 0.0
		for orthologs in X_orthologs:
			# An efficient way to compute the manifold part of the loss
			ortholog_predictions = np.dot(orthologs, w)
			loss += np.sum(species_graph_adjacency * \
						   (np.tile(ortholog_predictions,
									n_species).reshape(n_species, -1).T - \
									ortholog_predictions)**2)
		return loss

	def _objective(self, w, X, y, X_orthologs, species_graph_adjacency,
				   n_examples, n_features):
		return self._objective_negative_log_likelihood(w, X, y) / n_examples + \
			   self.alpha * self._objective_l2_norm(w) + \
			   self.beta * self._objective_orthologs(w, X_orthologs, species_graph_adjacency) / n_examples

	def _predict(self, X, w=None):
		if w is None:
			w = self.w
		return sigmoid(np.dot(X, w)).reshape(-1,)


if __name__ == "__main__":
	logging.basicConfig(level=logging.DEBUG,
								format="%(asctime)s.%(msecs)d %(levelname)s %(module)s - %(funcName)s: %(message)s")

	np.random.seed(42)
	X = np.random.rand(100, 5)
	y = np.random.randint(0, 2, 100).astype(np.float)

	A = np.random.rand(100, 100)
	A = np.dot(A.T, A)

	clf = AutogradLogisticRegression(alpha=1e-9, beta=1e-9, opti_lr=1e-1, opti_max_epochs=500)
	clf.fit(X=X,
			X_species=[0] * len(y),
			y=y,
			orthologs=dict([(i, {"species": range(100), "X": np.random.rand(100, 5)}) for i in xrange(X.shape[0])]),
			species_graph_adjacency=A,
			species_graph_names=range(100))

	from sklearn.metrics import roc_auc_score
	print roc_auc_score(y_score=clf.predict(X), y_true=y)
	exit()
