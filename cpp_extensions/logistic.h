#ifndef OPTIMIZED_GRADIENTS_H
#define OPTIMIZED_GRADIENTS_H

#include <unordered_map>

int compute_objective(double *X,
                      double *y,
                      std::unordered_map<int, double*> ortholog_features,
                      double *species_adjacency,
                      double *w,
                      int n_species,
                      int n_features,
                      int n_examples,
                      double &out_likelihood,
                      double &out_l2norm,
                      double &out_phylo);

int compute_sgd_gradient(double *X,  // Example features
                 double *y,  // Example labels
                 int iteration_example_idx,  // The index of the examples for this SDG iteration
                 double *species_adjacency,
                 double *ortholog_features,
                 double *w,  // The coefficient vector
                 int n_species,
                 int n_features,
                 int n_examples,
                 double *out_likelihood_gradient,
                 double *out_l2norm_gradient,
                 double *out_phylo_gradient);

#endif //OPTIMIZED_GRADIENTS_H
