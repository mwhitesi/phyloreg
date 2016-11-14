"""


"""
import numpy as np

class BaseAdjacencyMatrixBuilder(object):
    """


    """
    def __init__(self):
        pass

    def __call__(self, tree, type="adjacency"):
        """
        Compute the distance matrix between species based on the phylogenetic tree

        Parameters:
        -----------
        tree: many possible formats
            The phylogenetic tree of species
        type: str
            The format of the phylogenetic tree (adjacency, newick)

        Return:
        -------
        species: array_like, shape=(n_species,), dtype=str
            A list of species names in the same order as the dimensions of the distance matrix
        distance: array_like, shape=(n_species, n_species), dtype=float

        """
        return self._distance_func(tree, type)


# TODO: implement the adjacency matrix builder
class CommonAncestorAdjacencyMatrixBuilder(BaseAdjacencyMatrixBuilder):
    """
    Only consider the distance between two species if they are connected via a common ancestor in the tree

    """
    def __init__(self, sigma=1.0):
        """


        """
        self.sigma = float(sigma)
        super(CommonAncestorAdjacencyMatrixBuilder, self).__init__()

    def _distance_func(self, tree, type):
        return np.array([""] * 4), np.random.randint(0, 100, (4, 4))