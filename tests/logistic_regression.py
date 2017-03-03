"""
Testing the logistic regression implementation

"""
import logging
import numpy as np

from phyloreg.classifiers import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.DEBUG,
                                format="%(asctime)s.%(msecs)d %(levelname)s %(module)s: %(message)s")

np.random.seed(1)
n_examples = 1000
n_features = 4000
n_orthologs = 0
ortholog_names = [str(x) for x in np.random.randint(0, 1000000, n_orthologs)]

A = np.random.rand(n_orthologs + 1, n_orthologs + 1)
A = np.dot(A, A)
np.fill_diagonal(A, A.max())

X, y = make_classification(n_samples=n_examples, n_features=n_features)

clf = LogisticRegression(alpha=1e-3, beta=0.0, fit_intercept=True, opti_learning_rate=1e-1, opti_max_iter=1000)
clf.fit(X=X,
        X_species=["a"] * n_examples,
        y=y,
        orthologs={i: {"species": ortholog_names,
                       "X": np.array(np.random.randint(0, 5, (n_orthologs, n_features)))} for i in xrange(n_examples)},
        species_graph_adjacency=A,
        species_graph_names=["a"] + ortholog_names)

print "AUC:", roc_auc_score(y_true=y, y_score=clf.predict(X))