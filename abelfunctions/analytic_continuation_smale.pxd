from .analytic_continuation cimport AnalyticContinuator
from .polynomials cimport MultivariatePolynomial

cdef class AnalyticContinuatorSmale(AnalyticContinuator):
    cdef MultivariatePolynomial[:] df
