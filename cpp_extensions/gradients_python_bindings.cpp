#include <Python.h>
#include <numpy/arrayobject.h>

#include <cmath>
#include <iostream>

#include "gradients.h"

static PyObject *
t3_gradient(PyObject *self, PyObject *args){
    PyArrayObject *species_adjacency, *ortholog_features, *w; //borrowed

    // Extract the argument values
    if(!PyArg_ParseTuple(args, "O!O!O!",
                         &PyArray_Type, &species_adjacency,
                         &PyArray_Type, &ortholog_features,
                         &PyArray_Type, &w
    )){
        return NULL;
    }

    // Check the data types of the numpy arrays
    if(PyArray_TYPE(species_adjacency)!=PyArray_DOUBLE){
        PyErr_SetString(PyExc_TypeError,
                        "species_adjacency must be numpy.ndarray type double");
        return NULL;
    }
    if(PyArray_TYPE(ortholog_features)!=PyArray_DOUBLE){
        PyErr_SetString(PyExc_TypeError,
                        "ortholog_features must be numpy.ndarray type double");
        return NULL;
    }
    if(PyArray_TYPE(w)!=PyArray_DOUBLE){
        PyErr_SetString(PyExc_TypeError,
                        "w must be numpy.ndarray type double");
        return NULL;
    }

    // Check the dimensions of the numpy arrays
    npy_intp sa_dim0 = PyArray_DIM(species_adjacency,0);
    npy_intp sa_dim1 = PyArray_DIM(species_adjacency,1);
    npy_intp of_dim0 = PyArray_DIM(ortholog_features,0);
    npy_intp of_dim1 = PyArray_DIM(ortholog_features,1);
    npy_intp w_dim0 = PyArray_DIM(w,0);
    if(sa_dim0 != sa_dim1){
        PyErr_SetString(PyExc_ValueError,
                        "species_adjacency must be a square matrix (also symmetric)");
        return NULL;
    }
    if(sa_dim0 != of_dim0){
        PyErr_SetString(PyExc_ValueError,
                        "species_adjacency and ortholog_features must have the same number of rows");
        return NULL;
    }
    if(w_dim0 != of_dim1){
        PyErr_SetString(PyExc_ValueError,
                        "w must have the same number of elements than the number of rows in ortholog_features");
        return NULL;
    }

    // Access the array data
    double *species_adjacency_data = (double*)PyArray_DATA(species_adjacency);
    double *ortholog_features_data = (double*)PyArray_DATA(ortholog_features);
    double *w_data = (double*)PyArray_DATA(w);

    // Initialize arrays for return
    PyObject *gradient = PyArray_SimpleNew(1, &of_dim1, PyArray_DOUBLE);
    double *gradient_data = (double*)PyArray_DATA(gradient);

    int status = t3_gradient(species_adjacency_data, ortholog_features_data, w_data, of_dim0, of_dim1, gradient_data);
    if(status != 0){
        return NULL;
    }

    return Py_BuildValue("N", gradient);
}

static PyMethodDef Methods[] = {
        {"t3_gradient", t3_gradient, METH_VARARGS,
                "Computes the gradient of the manifold term in the logistic regression objective function."},
        {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initgradients
        (void){
    (void)Py_InitModule("gradients", Methods);
    import_array();//necessary from numpy otherwise we crash with segfault
}