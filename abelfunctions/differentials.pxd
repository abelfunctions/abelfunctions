from .polynomials cimport MultivariatePolynomial
from .riemann_surface_path cimport RiemannSurfacePathPrimitive

cdef class Differential:
    cdef object RS
    cdef object x
    cdef object y
    cdef object numer
    cdef object denom
    cdef MultivariatePolynomial numer_n
    cdef MultivariatePolynomial denom_n
    cpdef complex eval(self, complex, complex)
    cpdef complex[:] evaluate(self, RiemannSurfacePathPrimitive, double[:])

cdef class AbelianDifferentialFirstKind(Differential):
    pass

cdef class AbelianDifferentialSecondKind(Differential):
    pass

