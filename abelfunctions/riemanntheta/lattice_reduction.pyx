#Grady Williams
#Cython program provides a wrapper for a C-functions which performs the 
#LLL lattice reduction algorithm

cimport cython
import numpy as np
cimport numpy as np
np.import_array()

cdef extern from 'lll_reduce.h':
     void lll_reduce(double *, int, double, double)
     void gram_schmidt(double*, double*, double*, int)

@cython.boundscheck(False)
def lattice_reduce(T):
    dim = len(T)
    b = np.copy(T)
    b = np.ascontiguousarray(b, dtype = 'double')
    lll_reduce(<double*> np.PyArray_DATA(b),
	       dim, .50, .75)
    return b

def gram_schmidt_cy(b):
    dim = len(b)
    mu = np.zeros((dim,dim))
    B = np.zeros(dim)
    b = np.ascontiguousarray(b, dtype = 'double')
    B = np.ascontiguousarray(B, dtype = 'double')
    mu = np.ascontiguousarray(mu, dtype = 'double')
    gram_schmidt(<double*> np.PyArray_DATA(b),
                <double*> np.PyArray_DATA(mu),
		<double*> np.PyArray_DATA(B),
		dim)

 

    