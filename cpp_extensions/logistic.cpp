#include <cmath>
#include <time.h>       /* time */
#include <iostream>
#include "logistic.h"


inline double predict(double *w, double *X, int example_idx, int n_features, bool sigmoid){
    double dot = 0.0;
    for(int i = 0; i < n_features; i++){
        dot += X[example_idx * n_features + i] * w[i];
    }

    if(sigmoid)
        return 1.0 / (1.0 + exp(-dot));
    else
        return dot;
}



int compute_sgd_gradient(double *X, double *y, int iteration_example_idx, double *species_adjacency, double *ortholog_features, double *w, int n_species,
                 int n_features, int n_examples, double *out_likelihood_gradient, double *out_l2norm_gradient, double *out_phylo_gradient){

    // Initialize the output gradients
    for(int i = 0; i < n_features; i++){
        out_likelihood_gradient[i] = 0;
        out_l2norm_gradient[i] = 0;
        out_phylo_gradient[i] = 0;
    }

    // Gradient of the likelihood term
    double c1 = y[iteration_example_idx] - predict(w, X, iteration_example_idx, n_features, true);
    for(int i = 0; i < n_features; i++){
        out_likelihood_gradient[i] = X[iteration_example_idx * n_features + i] * c1;
    }

    // Gradient of the l2 norm term
    for(int i = 0; i < n_features; i++){
        out_l2norm_gradient[i] = 2.0 * w[i];
    }

    // Gradient of the phylo regularization term
    double *p = new double[n_species];
    double *top = new double[n_species];
    for(int i = 0; i < n_species; i++){
        top[i] = exp(-predict(w, ortholog_features, i, n_features, false));
        p[i] = predict(w, ortholog_features, i, n_features, true);
    }
    for(int t = 0; t < n_features; t++) {
        for(int k = 0; k < n_species; k++){
            for(int l = 0; l < n_species; l++){
                if(k < l){
                    out_phylo_gradient[t] += species_adjacency[k * n_species + l] * (p[k] - p[l]) * (p[k] * p[k] * top[k] * ortholog_features[k * n_features + t] - p[l] * p[l] * top[l] * ortholog_features[l * n_features + t]);
                }
            }
        }
        out_phylo_gradient[t] *= 4.0 / (n_species * n_species);
    }

    return 0;
}



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
                       double &out_phylo){

    // Initialize objective values
    out_likelihood = 0.0;
    out_l2norm = 0.0;
    out_phylo = 0.0;

    // Likelihood term
    for(int i = 0; i < n_examples; i++){
        double pi = predict(w, X, i, n_features, true);
        if(y[i] == 0)
            pi = 1.0 - pi;
        out_likelihood += log(pi);
    }
    out_likelihood /= n_examples;

    // L2 norm term
    for(int i = 0; i < n_features; i++){
        out_l2norm += w[i] * w[i];
    }

    // Phylo regularization term
    for(int i = 0; i < n_examples; i++){
        // Compute the prediction for each ortholog
        double *p = new double[n_species];
        for(int j = 0; j < n_species; j++){
            p[j] = predict(w, ortholog_features.find(i)->second, j, n_features, true);
        }

        // Compute the weighted difference between the ortholog predictions
        for(int k = 0; k < n_species; k++){
            for(int l = 0; l < k; l++){
                double diff = p[k] - p[l];
                out_phylo += 2.0 * species_adjacency[k * n_species + l] * (diff * diff);
            }
        }
    }
    out_phylo /= n_examples * n_species * n_species;

    return 0;
}