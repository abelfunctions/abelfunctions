from abelfunctions.riemann_surface cimport RiemannSurface
from abelfunctions.riemann_surface_path cimport RiemannSurfacePathPrimitive

cdef class AnalyticContinuator:
    cdef RiemannSurface RS
    cdef int deg
    cpdef complex[:] analytically_continue(self, RiemannSurfacePathPrimitive, complex, complex[:], complex)
