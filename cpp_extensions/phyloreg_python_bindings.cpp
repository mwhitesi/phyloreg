#include <Python.h>
#include <numpy/arrayobject.h>

#include <cmath>
#include <iostream>
#include <unordered_map>
#include <utility>

#include "logistic.h"


double *example_features, *example_labels, *species_adjacency;
std::unordered_map<int, double*> example_orthologs;
int n_examples = -1;
int n_features = -1;
int n_species = -1;
bool all_initialized = false;
bool examples_initialized = false;
bool species_adjacency_initialized = false;

/***********************************************************************************************************************
 *                                              MODULE GLOBAL VARIABLES
 **********************************************************************************************************************/
static PyObject *
set_examples_and_labels(PyObject *self, PyObject *args){
    PyArrayObject *X, *y; //borrowed

    // Extract the argument values
    if(!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &X, &PyArray_Type, &y)){
        return NULL;
    }

    // Check the type of the numpy arrays
    if(PyArray_TYPE(X) != PyArray_DOUBLE){
        PyErr_SetString(PyExc_TypeError,
                        "the example feature matrix must be numpy.ndarray type double");
        return NULL;
    }
    if(PyArray_TYPE(y) != PyArray_DOUBLE){
        PyErr_SetString(PyExc_TypeError,
                        "the label vector must be numpy.ndarray type double");
        return NULL;
    }

    // Check the dimensions of the numpy arrays
    if(PyArray_NDIM(X) != 2){
        PyErr_SetString(PyExc_TypeError,
                        "the example feature matrix must be a 2D numpy.ndarray");
        return NULL;
    }
    if(PyArray_NDIM(y) != 1){
        PyErr_SetString(PyExc_TypeError,
                        "the label vector must be a 1D numpy.ndarray");
        return NULL;
    }
    npy_intp X_dim0 = PyArray_DIM(X, 0);
    npy_intp X_dim1 = PyArray_DIM(X, 1);
    npy_intp y_dim0 = PyArray_DIM(y, 0);
    if(X_dim0 != y_dim0){
        PyErr_SetString(PyExc_TypeError,
                        "the example feature matrix and the label vector must have the same number of rows");
        return NULL;
    }
    if(n_features == -1)
    {
        n_features = (int) X_dim1;
    }
    else{
        if(X_dim1 != n_features){
            PyErr_SetString(PyExc_TypeError,
                            "the example feature matrix has the wrong number of features");
            return NULL;
        }
    }

    // Save the features and labels
    example_features = (double*)PyArray_DATA(PyArray_GETCONTIGUOUS(X));
    example_labels = (double*)PyArray_DATA(PyArray_GETCONTIGUOUS(y));
    examples_initialized = true;
    n_examples = X_dim0;
    all_initialized = false;

    return Py_None;
}


static PyObject *
set_orthologs(PyObject *self, PyObject *args){
    int example_idx;

    PyArrayObject *ortholog_features; //borrowed

    // Extract the argument values
    if(!PyArg_ParseTuple(args, "iO!", &example_idx, &PyArray_Type, &ortholog_features)){
        return NULL;
    }

    if(PyArray_TYPE(ortholog_features)!=PyArray_DOUBLE){
        PyErr_SetString(PyExc_TypeError,
                        "ortholog_features must be numpy.ndarray type double");
        return NULL;
    }

    // Check the dimensions of the numpy array
    npy_intp of_dim0 = PyArray_DIM(ortholog_features,0);
    npy_intp of_dim1 = PyArray_DIM(ortholog_features,1);
    if(n_features == -1){
        n_features = (int) of_dim1;
    }
    else{
        if(n_features != of_dim1){
            PyErr_SetString(PyExc_TypeError,
                            "ortholog feature matrix has wrong number of columns");
            return NULL;
        }
    }
    if(n_species == -1){
        n_species = (int) of_dim0;
    }
    else{
        if(n_species != of_dim0){
            PyErr_SetString(PyExc_TypeError,
                            "ortholog feature matrix has wrong number of rows");
            return NULL;
        }
    }

    // Access the array data
    double *ortholog_features_data = (double*)PyArray_DATA(PyArray_GETCONTIGUOUS(ortholog_features));

    // Save the ortholog matrix for the example
    example_orthologs.insert(std::pair<int, double*>(example_idx, ortholog_features_data));
    all_initialized = false;

    return Py_None;
}

static PyObject *
set_species_adjacency(PyObject *self, PyObject *args){
    PyArrayObject *sa; //borrowed

    // Extract the argument values
    if(!PyArg_ParseTuple(args, "O!", &PyArray_Type, &sa)){
        return NULL;
    }

    if(PyArray_TYPE(sa)!=PyArray_DOUBLE){
        PyErr_SetString(PyExc_TypeError,
                        "species_adjacency must be numpy.ndarray type double");
        return NULL;
    }

    // Check the dimensions of the numpy array
    npy_intp sa_dim0 = PyArray_DIM(sa,0);
    npy_intp sa_dim1 = PyArray_DIM(sa,1);
    if(sa_dim0 != sa_dim1){
        PyErr_SetString(PyExc_TypeError,
                        "species adjacency matrix must be symmetric");
        return NULL;
    }
    if(n_species == -1){
        n_species = (int) sa_dim0;
    }
    else{
        if(n_species != sa_dim0){
            PyErr_SetString(PyExc_TypeError,
                            "species adjacency matrix has wrong number of rows");
            return NULL;
        }
    }

    // Access the array data
    species_adjacency = (double*)PyArray_DATA(PyArray_GETCONTIGUOUS(sa));
    species_adjacency_initialized = true;
    all_initialized = false;

    return Py_None;
}

static PyObject *
reset(PyObject *self, PyObject *args){
    PyArrayObject *sa; //borrowed

    // Extract the argument values
    if(!PyArg_ParseTuple(args, "", &PyArray_Type, &sa)){
        return NULL;
    }

    example_orthologs.clear();
    n_examples = -1;
    n_features = -1;
    n_species = -1;
    all_initialized = false;
    examples_initialized = false;
    species_adjacency_initialized = false;

    return Py_None;
}



/***********************************************************************************************************************
 *                                                  GRADIENTS AND OBJECTIVES
 **********************************************************************************************************************/
static PyObject *
get_gradient(PyObject *self, PyObject *args){

    // Check that we have the orthologs for all examples
    if(!all_initialized){
        bool all_orthologs_initialized = example_orthologs.size() == n_examples;
        for (int i = 0; i < n_examples && all_orthologs_initialized; i++) {
            all_orthologs_initialized =
                    all_orthologs_initialized && (example_orthologs.find(i) != example_orthologs.end());
        }

        if (!all_orthologs_initialized || !examples_initialized || !species_adjacency_initialized) {
            PyErr_SetString(PyExc_TypeError,
                            "Solver is not completely initialized. The example features, labels and orthologs must be set, "
                                    "as well as the species adjacency matrix.");
            return NULL;
        } else
            all_initialized = true; // save this result for the next iterations (its long to compute)
    }

    int sgd_iteration_example;
    PyArrayObject *w; //borrowed

    // Extract the argument values
    if(!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &w, &sgd_iteration_example
    )){
        return NULL;
    }

    // Check the bounds of the example idx
    if(sgd_iteration_example < 0 || sgd_iteration_example > n_examples - 1){
        PyErr_SetString(PyExc_TypeError,
                        "the example to use for the stochastic gradient descent iterator must have an index between 0 and n_examples");
        return NULL;
    }

    // Check the type of the numpy arrays
    if(PyArray_TYPE(w)!=PyArray_DOUBLE){
        PyErr_SetString(PyExc_TypeError,
                        "w must be numpy.ndarray type double");
        return NULL;
    }

    // Check the dimensions of the numpy arrays
    if(PyArray_NDIM(w) != 1){
        PyErr_SetString(PyExc_ValueError,
                        "w must be a vector (1D numpy.ndarray)");
        return NULL;
    }
    npy_intp w_dim0 = PyArray_DIM(w,0);
    if(w_dim0 != n_features){
        PyErr_SetString(PyExc_ValueError,
                        "w must have as many elements as there are features");
        return NULL;
    }

    // Access the array data
    double *w_data = (double*)PyArray_DATA(w);

    // Initialize arrays for return
    PyObject *likelihood_gradient = PyArray_SimpleNew(1, &w_dim0, PyArray_DOUBLE);
    double *likelihood_gradient_data = (double*)PyArray_DATA(likelihood_gradient);
    PyObject *l2norm_gradient = PyArray_SimpleNew(1, &w_dim0, PyArray_DOUBLE);
    double *l2norm_gradient_gradient_data = (double*)PyArray_DATA(l2norm_gradient);
    PyObject *phylo_gradient = PyArray_SimpleNew(1, &w_dim0, PyArray_DOUBLE);
    double *phylo_gradient_gradient_data = (double*)PyArray_DATA(phylo_gradient);

    // Compute the gradient
    int status = compute_sgd_gradient(example_features, example_labels, sgd_iteration_example, species_adjacency, example_orthologs.find(sgd_iteration_example)->second, w_data, n_species, n_features, n_examples, likelihood_gradient_data, l2norm_gradient_gradient_data, phylo_gradient_gradient_data);
    if(status != 0){
        return NULL;
    }

    return Py_BuildValue("N,N,N", likelihood_gradient, l2norm_gradient, phylo_gradient);
}

static PyObject *
get_objective(PyObject *self, PyObject *args){

    // Check that we have the orthologs for all examples
    bool all_orthologs_initialized = example_orthologs.size() == n_examples;
    for(int i = 0; i < n_examples && all_orthologs_initialized; i++){
        all_orthologs_initialized = all_orthologs_initialized && (example_orthologs.find(i) != example_orthologs.end());
    }

    if(!all_orthologs_initialized || !examples_initialized || !species_adjacency_initialized){
        PyErr_SetString(PyExc_TypeError,
                        "Solver is not completely initialized. The example features, labels and orthologs must be set, "
                                "as well as the species adjacency matrix.");
        return NULL;
    }

    PyArrayObject *w; //borrowed

    // Extract the argument values
    if(!PyArg_ParseTuple(args, "O!", &PyArray_Type, &w
    )){
        return NULL;
    }

    // Check the type of the numpy arrays
    if(PyArray_TYPE(w)!=PyArray_DOUBLE){
        PyErr_SetString(PyExc_TypeError,
                        "w must be numpy.ndarray type double");
        return NULL;
    }

    // Check the dimensions of the numpy arrays
    if(PyArray_NDIM(w) != 1){
        PyErr_SetString(PyExc_ValueError,
                        "w must be a vector (1D numpy.ndarray)");
        return NULL;
    }
    npy_intp w_dim0 = PyArray_DIM(w,0);
    if(w_dim0 != n_features){
        PyErr_SetString(PyExc_ValueError,
                        "w must have as many elements as there are features");
        return NULL;
    }

    // Access the array data
    double *w_data = (double*)PyArray_DATA(w);

    // Initialize variables for return
    double likelihood = -INFINITY;
    double l2norm = -INFINITY;
    double phylo = -INFINITY;

    // Compute the gradient
    int status = compute_objective(example_features, example_labels, example_orthologs, species_adjacency, w_data, n_species, n_features, n_examples, likelihood, l2norm, phylo);
    if(status != 0){
        return NULL;
    }

    return Py_BuildValue("ddd", likelihood, l2norm, phylo);
}



/***********************************************************************************************************************
 *                                                  MODULE DECLARATION
 **********************************************************************************************************************/
static PyMethodDef Methods[] = {
        {"get_gradient", get_gradient, METH_VARARGS,
                "Computes the gradient of the objective function."},
        {"get_objective", get_objective, METH_VARARGS,
                "Computes the value of the objective function."},
        {"set_examples_and_labels", set_examples_and_labels, METH_VARARGS,
                "Defines the feature vectors and labels for the learning examples."},
        {"set_example_orthologs", set_orthologs, METH_VARARGS,
                "Defines the ortholog feature matrix of a learning example."},
        {"set_species_adjacency", set_species_adjacency, METH_VARARGS,
                "Defines the species adjacency matrix."},
        {"reset", reset, METH_VARARGS,
                "Resets the solver."},
        {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
init_phyloreg
        (void){
    (void)Py_InitModule("_phyloreg", Methods);
    import_array();//necessary from numpy otherwise we crash with segfault
}