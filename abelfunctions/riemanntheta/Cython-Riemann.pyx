#Grady Williams (gradyrw@uw.edu) - October 2012

#This Cython program provides a 

cimport cython
import numpy as np
cimport numpy as np
np.import_array()

cdef extern from 'riemanntheta.h':
        double finite_sum_without_derivatives(double *, double *, double *,
					      double *, double *, double *,
					      double *, double *, int, int)

#Creates a Python-accessible function in cython that calls the 
#C - function above

@cython.boundscheck(False) #Turns off bounds checking
def finite_sum(X, Yinv, T, x, y, S, g):
        N = len(S)/g
        cdef double real[0]
        cdef double imag[0]
        X = np.ascontiguousarray(X, dtype = 'double')
        Yinv = np.ascontiguousarray(Yinv, dtype = 'double')
        x = np.ascontiguousarray(x, dtype = 'double')
        y = np.ascontiguousarray(y, dtype = 'double')
        S = np.ascontiguousarray(S, dtype = 'double')
        finite_sum_without_derivatives(real, imag,
                                       <double*> np.PyArray_DATA(X),
                                       <double*> np.PyArray_DATA(Yinv),
                                       <double*> np.PyArray_DATA(T),
                                       <double*> np.PyArray_DATA(x),
                                       <double*> np.PyArray_DATA(y),
                                       <double*> np.PyArray_DATA(S),
                                        g, N)
        I = 1.0j
        return real[0] + imag[0]*1.0j
                 
