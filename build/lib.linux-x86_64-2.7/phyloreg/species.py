"""


"""
import numpy as np

from collections import defaultdict
from math import exp

class BaseAdjacencyMatrixBuilder(object):
    """


    """
    def __init__(self):
        pass

    def __call__(self, tree):
        """
        Compute the distance matrix between species based on the phylogenetic tree

        Parameters:
        -----------
        tree: many possible formats
            The phylogenetic tree of species, which is a dictionnary where the keys are species and the values are the
            name of the direct ancestor and the distance to it. (None, None) indicates that this species is the root of
            the tree.

            Ex: {"human": ("ancestor1", 10),
                 "monkey": ("ancestor1", 100),
                 "fly": ("ancestor2", 32),
                 "alien": ("ancestor2", 1000),
                 "ancestor1": ("ancestor3", 2),
                 "ancestor2": ("ancestor3", 1),
                 "ancestor3": (None, None)}

        Return:
        -------
        species: array_like, shape=(n_species,), dtype=str
            A list of species names in the same order as the dimensions of the distance matrix
        distance: array_like, shape=(n_species, n_species), dtype=float

        """
        return self._build_adjacency(tree)


class ExponentialAdjacencyMatrixBuilder(BaseAdjacencyMatrixBuilder):
    """
    Species that are directly connected in the tree will have similarity exp(-((distance/sigma)**2)).
    Species that are not directly connected in the tree will have similarity 0.

    """
    def __init__(self, sigma=1.0):
        self.sigma = float(sigma)
        super(ExponentialAdjacencyMatrixBuilder, self).__init__()

    def _build_adjacency(self, tree):
        A = np.zeros((len(tree), len(tree)))
        species_idx = dict(zip(tree.keys(), np.arange(len(tree))))
        for i, (species, (ancestor, distance)) in enumerate(tree.iteritems()):
            A[i, i] = 1
            if ancestor is None:
                continue  # Node is the root
            A[i, species_idx[ancestor]] = A[species_idx[ancestor], i] = exp(-distance**2 / (2 * self.sigma**2))
        return tree.keys(), A


# class CommonAncestorAdjacencyMatrixBuilder(BaseAdjacencyMatrixBuilder):
#     """
#     Only consider the distance between two species if they are connected via a common ancestor in the tree
#
#     """
#     def __init__(self, sigma=1.0):
#         """
#
#
#         """
#         self.sigma = float(sigma)
#         super(CommonAncestorAdjacencyMatrixBuilder, self).__init__()
#
#     def _build_adjacency(self, tree, tree_type):
#         if tree_type == "distance":
#             # The tree is provided as an distance matrix (species x species)
#             B = np.zeros(A.shape)
#
#             # Find the root
#             root_candidates = [np.sum(x != 0) == 2 for x in A]
#             assert sum(root_candidates) == 1
#
#             roots = [np.where(root_candidates)[0][0]]
#             seen = defaultdict(bool)
#             while len(roots) > 0:
#                 root_idx = roots.pop()
#                 seen[root_idx] = True
#                 print "Current root is", root_idx + 1
#                 children_idx = [x for x in np.where(A[root_idx] != 0)[0] if not seen[x]]
#                 print "Its children are:", [x + 1 for x in children_idx]
#
#                 B[children_idx] = 1
#
#                 roots += children_idx
#
#             print B
#
#         else:
#             raise NotImplementedError("Unsupported tree type.")