#Grady Williams (gradyrw@uw.edu) - October 2012

#This Cython program provides a wrapper for C-functions which compute the finitesum of the Riemann Theta finite sum approximation

cimport cython
import numpy as np
cimport numpy as np
np.import_array()
import scipy.linalg as la

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
    T = np.ascontiguousarray(T, dtype = 'double')
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
def finite_sum_derivatives(X, Yinv, T, x, y, S, deriv, g):
    N = len(S)/g
    nderivs = len(deriv)
    cdef double real[0]
    cdef double imag[0]
    deriv_real = np.ascontiguousarray(deriv.real, dtype = 'double')
    deriv_imag = np.ascontiguousarray(deriv.imag, dtype = 'double')
    X = np.ascontiguousarray(X, dtype = 'double')
    Yinv = np.ascontiguousarray(Yinv, dtype = 'double')
    T = np.ascontiguousarray(T, dtype = 'double')
    x = np.ascontiguousarray(x, dtype = 'double')
    y = np.ascontiguousarray(y, dtype = 'double')
    S = np.ascontiguousarray(S, dtype = 'double')
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
  
def find_int_points(g, c, R, T):
    Tinv = la.inv(T)
    points = []
    stack = []
    stack.append([[], g, c, R])
    FINISHED = False
    while (not FINISHED):
        N = stack.pop()
        start = N[0]
        g = N[1]
        c = N[2]
        R = N[3]
        a = int( np.ceil((c[g] - R/T[g,g]).real) )
        b = int( np.floor((c[g] + R/T[g,g]).real) )
        #Check if reached the edge of the ellipsoid
        if not a < b: continue
        #Last dimension reached, append points
        if g == 0:
            for x in range(a, b+1):
                s = start[:]
                s.append(x)
                s.reverse()
                points.extend(s)
        else:
            for x in range(a, b+1):
                chat = c[g]
                that = T[:g,g]
                newc = chat - Tinv[:g, :g] * that * (x - c[g])
                newR = np.sqrt(R**2 - (T[g,g] * (x - c[g]))**2)
                newStart = start[:]
                newStart.append(x)
                stack.append([newStart, g - 1, newc, newR[0]])
        if (len(stack) == 0):
            FINISHED = True
    return points

                 
