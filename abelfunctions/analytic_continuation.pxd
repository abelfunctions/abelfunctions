from abelfunctions.differentials cimport Differential
from abelfunctions.riemann_surface cimport RiemannSurface
from abelfunctions.riemann_surface_path cimport RiemannSurfacePathPrimitive

cdef class AnalyticContinuator:
    cdef RiemannSurface RS
    cdef RiemannSurfacePathPrimitive gamma
    cdef int deg
    cpdef complex[:] analytically_continue(self, complex, complex[:], complex)
    cpdef complex integrate(self, Differential)

cdef class AnalyticContinuatorPuiseux(AnalyticContinuator):
    cdef object center
    cdef object puiseux_series
    cdef object _target_place


