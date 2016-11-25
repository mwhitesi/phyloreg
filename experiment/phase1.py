import sys
sys.path.insert(0, '..')
from phyloreg import species
from treestructure import getTreeDict

# 100 Way alignment tree file name with full path
treefilename = 'modtree100'

# create tree dict
phylo = getTreeDict( treefilename )
print ( phylo )

# create adjacency matrix

#adj_matrix = species.BaseAdjacencyMatrixBuilder(  )

#print ( adj_matrix( phylo ) )

adj_matrix = species.ExponentialAdjacencyMatrixBuilder()

print ( adj_matrix( phylo ) )



