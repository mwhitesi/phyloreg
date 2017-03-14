//
// Created by Alexandre Drouin on 2017-03-14.
//
#include <cmath>
#include <iostream>
#include "gradients.h"

int t3_gradient(double *species_adjacency,
                 double *ortholog_features,
                 double *w,
                 int n_orthologs,
                 int n_features,
                 double *out_gradient){

    // Compute p and top
    double *p = new double[n_orthologs];
    double *top = new double[n_orthologs];
    for(int i = 0; i < n_orthologs; i++){
        double dot = 0.0;
        for(int j = 0; j < n_features; j++){
            dot += ortholog_features[i * n_features + j] * w[j];
        }
        top[i] = exp(-dot);
        p[i] = 1.0 / (1.0 + top[i]);
    }

    // Compute the gradient of each feature
    for(int t = 0; t < n_features; t++) {
        for(int k = 0; k < n_orthologs; k++){
            for(int l = 0; l < n_orthologs; l++){
                if(k < l){
                    out_gradient[t] += species_adjacency[k * n_orthologs + l] * (p[k] - p[l]) * (p[k] * p[k] * top[k] * ortholog_features[k * n_features + t] - p[l] * p[l] * top[l] * ortholog_features[l * n_features + t]);
                }
            }
        }
        out_gradient[t] *= 4.0 / (n_orthologs * n_orthologs);
    }

    return 0;
}