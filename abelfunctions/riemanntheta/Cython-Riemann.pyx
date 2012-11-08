#Grady Williams (gradyrw@uw.edu) - October 2012

#This Cython program provides a 

cimport cython
import numpy as np
cimport numpy as np
np.import_array()

cdef extern from 'riemanntheta.h':
        void finite_sum_without_derivatives(double *, double *, double *,
					                        double *, double *, double *,
					                        double *, double *, int, int)

        void finite_sum_with_derivatives(double*, double*, double*, double*, 
                                         double*, double*, double*, double*,
                                         double*, double*, int , int , int )

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
    return real[0] + imag[0]*1.0j

@cython.boundscheck(False) #Turns off bounds checking
def finite_sum_derivatives(X, Yinv, T, x, y, S, deriv_real, 
                                deriv_imag, nderivs, g):
    N = len(S)/g
    cdef double real[0]
    cdef double imag[0]
    X = np.ascontiguousarray(X, dtype = 'double')
    Yinv = np.ascontiguousarray(Yinv, dtype = 'double')
    x = np.ascontiguousarray(x, dtype = 'double')
    y = np.ascontiguousarray(y, dtype = 'double')
    S = np.ascontiguousarray(S, dtype = 'double')
    deriv_real = np.ascontiguousarray(deriv_real, dtype = 'double')
    deriv_imag = np.ascontiguousarray(deriv_imag, dtype = 'double')
    finite_sum_with_derivatives(real, imag,
                                <double*> np.PyArray_DATA(X),
                                <double*> np.PyArray_DATA(Yinv),
                                <double*> np.PyArray_DATA(T),
                                <double*> np.PyArray_DATA(x),
                                <double*> np.PyArray_DATA(y),
                                <double*> np.PyArray_DATA(S),
                                <double*> np.PyArray_DATA(deriv_real),
                                <double*> np.PyArray_DATA(deriv_imag),
                                nderivs, g, N)
    return real[0] + imag[0]*1.0j
  
                 
