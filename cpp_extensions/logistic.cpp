#include <cmath>
#include <time.h>       /* time */
#include <iostream>
#include "logistic.h"

int t3_gradient(double *species_adjacency,
                 double *ortholog_features,
                 double *w,
                 int n_species,
                 int n_features,
                 double *out_gradient){

    // Compute p and top
    double *p = new double[n_species];
    double *top = new double[n_species];
    for(int i = 0; i < n_species; i++){
        double dot = 0.0;
        for(int j = 0; j < n_features; j++){
            dot += ortholog_features[i * n_features + j] * w[j];
        }
        top[i] = exp(-dot);
        p[i] = 1.0 / (1.0 + top[i]);
    }

    // Compute the gradient of each feature
    for(int t = 0; t < n_features; t++) {
        for(int k = 0; k < n_species; k++){
            for(int l = 0; l < n_species; l++){
                if(k < l){
                    out_gradient[t] += species_adjacency[k * n_species + l] * (p[k] - p[l]) * (p[k] * p[k] * top[k] * ortholog_features[k * n_features + t] - p[l] * p[l] * top[l] * ortholog_features[l * n_features + t]);
                }
            }
        }
        out_gradient[t] *= 4.0 / (n_species * n_species);
    }

    return 0;
}

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

inline double loss_likelihood(double *w, double *X, int example_idx, double *y, int n_examples, int n_features){
            double l=0.0, pi=0.0, dot=0.0;

            for(int i=0; i<n_features; i++){
                dot += X[example_idx * n_features + i] * w[i];
            }

            pi = 1.0 / (1.0 + np.exp(-1 * dot ))
            l += log(pi) ? y[example_idx] == 1 : log(1.0 - pi);

            l /= n_examples;
            return l;

}

inline double loss_l2(double *w, double alpha, int n_examples, int n_features){
    double norm = 0.0;
    for(int i=0; i<n_features; i++){
        norm += abs( w[i] * w[i]);
        }
    return alpha * norm;
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
        out_likelihood_gradient[i] = X[iteration_example_idx * n_features + i] * c1 / n_examples;
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
        out_phylo_gradient[t] *= 4.0 / (n_examples * n_species * n_species);
    }

    return 0;
}

int check_sgd_gradient(){
/*Checks if all three gradient terms are correctly computed
*/
    // Initialize the output gradients
    for(int i = 0; i < n_features; i++){
        out_likelihood_gradient[i] = 0;
        out_l2norm_gradient[i] = 0;
        out_phylo_gradient[i] = 0;
    }

    // Generated data for likelihood term
    int n_examples = 10;
    int n_features = 5;
    double epsilon = pow(10,-6);
    double tol = pow(10, -8);
    int min=0, max=2;


    //# Initialization
    srand(time(NULL)); // Seed the time
    //int finalNum = rand()%(max-min+1)+min; // Generate the number, assign to variable.

    double X[n_examples, n_features], y[n_examples], w[n_features];

    for(int i=0; i<n_examples; i++){
        for(int j=0; j<n_features; j++){
            X[i][j] = rand()%(max-min+1)+min;
        }
        y[i] = rand()%(max-min+1)+min;
    }

    for(int i=0; i<n_features; i++){
        w[i] = rand();
    }

    int iteration_example_idx = 2;
    // Gradient of the likelihood term
    double c1 = y[iteration_example_idx] - predict(w, X, iteration_example_idx, n_features, true);
    for(int i = 0; i < n_features; i++){
        out_likelihood_gradient[i] = X[iteration_example_idx * n_features + i] * c1 / n_examples;
    }

    // Empirical gradient of the likelihood term
    //# Compute the empirical gradient estimate
    //# Check the gradient for each component of w
    double w_1[n_features], w_2[n_features];
    for(int i=0; i<n_features; i++){
        for(int j=0; j<n_features; j++){
            w_1[j] = w[j];
            w_2[j] = w[j];
        }
        w_1[i] += epsilon;
        w_2[i] -= epsilon;
        empirical_gradient = (loss_likelihood(w_1, X, iteration_example_idx, y, n_examples, n_features)
                             - loss_likelihood(w_2, X, iteration_example_idx, y, n_examples, n_features))
                             / (2 * epsilon);

        if(abs(empirical_gradient - gradient[i]) > tol){
            cout<<"FAILED. Expected gradient: "<< empirical_gradient<<" Calculated gradient: "<< gradient[i];
            }


        }

    cout<<"PASSED. likelihood term";

    // Generate data for l2 norm term
    n_features = 5;
    alpha = rand() * 100.0;

//    # Initialization
    for(int i=0; i<n_features; i++){
            w[i] = rand();
        }

    // Gradient of the l2 norm term
    for(int i = 0; i < n_features; i++){
        out_l2norm_gradient[i] = 2.0 * alpha * w[i];
    }

    // Empirical gradient of the l2 norm term
//    # Compute the empirical gradient estimate
    double w_1[n_features], w_2[n_features];
    for(int i=0; i<n_features; i++){
        for(int j=0; j<n_features; j++){
            w_1[j] = w[j];
            w_2[j] = w[j];
        }
        w_1[i] += epsilon;
        w_2[i] -= epsilon;
        empirical_gradient = (loss_l2(w_1, alpha)
                             - loss_l2(w_2, alpha))
                             / (2 * epsilon);

        if(abs(empirical_gradient - gradient[i]) > tol){
            cout<<"FAILED. Expected gradient: "<< empirical_gradient<<" Calculated gradient: "<< gradient[i];
            }
        }

    cout<<"PASSED. l2 term";

//    // Generate data for phylo regularization term
//
//    // Gradient of the phylo regularization term
//    double *p = new double[n_species];
//    double *top = new double[n_species];
//    for(int i = 0; i < n_species; i++){
//        top[i] = exp(-predict(w, ortholog_features, i, n_features, false));
//        p[i] = predict(w, ortholog_features, i, n_features, true);
//    }
//    for(int t = 0; t < n_features; t++) {
//        for(int k = 0; k < n_species; k++){
//            for(int l = 0; l < n_species; l++){
//                if(k < l){
//                    out_phylo_gradient[t] += species_adjacency[k * n_species + l] * (p[k] - p[l]) * (p[k] * p[k] * top[k] * ortholog_features[k * n_features + t] - p[l] * p[l] * top[l] * ortholog_features[l * n_features + t]);
//                }
//            }
//        }
//        out_phylo_gradient[t] *= 4.0 / (n_examples * n_species * n_species);
//    }

    // Empirical gradient of the phylo regularization term

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
        out_phylo /= n_examples * n_species * n_species;
    }

    return 0;
}