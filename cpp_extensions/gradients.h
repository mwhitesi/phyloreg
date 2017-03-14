//
// Created by Alexandre Drouin on 2017-03-14.
//

#ifndef OPTIMIZED_GRADIENTS_H
#define OPTIMIZED_GRADIENTS_H

int t3_gradient(double *species_adjacency,
                double *ortholog_features,
                double *w,
                int n_orthologs,
                int n_features,
                double *out_gradient);

#endif //OPTIMIZED_GRADIENTS_H
