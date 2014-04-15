from abelfunctions.riemann_surface cimport RiemannSurface
from abelfunctions.analytic_continuation cimport AnalyticContinuator

cdef class RiemannSurfacePathPrimitive:
    cdef RiemannSurface _RS
    cdef AnalyticContinuator _AC
    cdef complex _x0
    cdef complex[:] _y0
    cdef RiemannSurfacePathPrimitive[:] _segments
    cdef int _nsegments
    cdef int _ncheckpoints
    cdef double[:] _tcheckpoints
    cdef complex[:] _xcheckpoints
    cdef complex[:,:] _ycheckpoints
    cdef int _nearest_checkpoint_index(self, double)
    cpdef complex get_x(self, double)
    cpdef complex get_dxdt(self, double)
    cpdef complex[:] analytically_continue(RiemannSurfacePathPrimitive,
                                           complex, complex[:], complex)
    cpdef complex[:] get_y(self, double)


cdef class RiemannSurfacePathLine(RiemannSurfacePathPrimitive):
    cdef complex z0
    cdef complex z1

cdef class RiemannSurfacePathArc(RiemannSurfacePathPrimitive):
    cdef complex R
    cdef complex w
    cdef complex theta
    cdef complex dtheta


cdef class RiemannSurfacePath(RiemannSurfacePathPrimitive):
    cdef int _get_segment_index(self, double)
    pass

