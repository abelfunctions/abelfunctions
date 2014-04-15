from .polynomials cimport MultivariatePolynomial

cdef class Differential:
    cdef object _omega
    cdef MultivariatePolynomial numer
    cdef MultivariatePolynomial denom
    cpdef complex eval(self, complex, complex)
