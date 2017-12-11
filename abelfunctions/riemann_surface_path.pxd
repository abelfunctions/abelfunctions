from sage.ext.interpreters.wrapper_el cimport Wrapper_el
from abelfunctions.complex_path cimport ComplexPathPrimitive

cdef class RiemannSurfacePathPrimitive:
    cdef object _riemann_surface
    cdef RiemannSurfacePathPrimitive[:] _segments
    cdef int _nsegments
    cdef ComplexPathPrimitive _complex_path
    cdef complex _x0
    cdef object[:] _y0
    cdef int _ncheckpoints
    cdef double[:] _scheckpoints
    cdef object[:] _xcheckpoints
    cdef object[:,:] _ycheckpoints
    cdef object _repr

    cpdef int _nearest_checkpoint_index(self, double s)
    cpdef object get_x(self, double s)
    cpdef object get_dxds(self, double s)
    cpdef object[:] get_y(self, double s)
    cpdef object[:] analytically_continue(self, object xi, object[:] yi, object xip1)

cdef class RiemannSurfacePath(RiemannSurfacePathPrimitive):
    cdef int segment_index_at_parameter(self, double s)

cdef class RiemannSurfacePathPuiseux(RiemannSurfacePathPrimitive):
    cdef object puiseux_series
    cdef object target_point
    cdef object target_place

cdef class RiemannSurfacePathSmale(RiemannSurfacePathPrimitive):
    cdef int _degree
    cdef Wrapper_el[:] _df
