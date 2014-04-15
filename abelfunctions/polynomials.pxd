cdef class UnivariatePolynomial:
    cdef int deg
    cdef complex[:] c
    cdef complex eval(self, complex) nogil

cdef class MultivariatePolynomial:
    cdef int deg
    cdef UnivariatePolynomial[:] c
    cdef complex eval(self, complex, complex)
