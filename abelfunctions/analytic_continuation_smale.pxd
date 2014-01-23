from abelfunctions.riemann_surface cimport RiemannSurface
from abelfunctions.analytic_continuation cimport AnalyticContinuator

cdef class UnivariatePolynomial:
    cdef int deg
    cdef complex[:] c
    cdef complex eval(self, complex) nogil

cdef class MultivariatePolynomial:
    cdef int deg
    cdef UnivariatePolynomial[:] c
    cdef complex eval(self, complex, complex)

cdef class AnalyticContinuatorSmale(AnalyticContinuator):
    cdef MultivariatePolynomial[:] df
