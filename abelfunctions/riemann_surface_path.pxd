from sage.ext.interpreters.wrapper_el cimport Wrapper_el
from abelfunctions.complex_path cimport ComplexPathPrimitive

cdef class RiemannSurfacePathPrimitive:
    cdef object _riemann_surface
    cdef RiemannSurfacePathPrimitive[:] _segments
    cdef int _nsegments
    cdef ComplexPathPrimitive _complex_path
    cdef complex _x0
    cdef complex[:] _y0
    cdef int _ncheckpoints
    cdef double[:] _scheckpoints
    cdef complex[:] _xcheckpoints
    cdef complex[:,:] _ycheckpoints
    cdef object _repr

    cpdef int _nearest_checkpoint_index(self, double s)
    cpdef complex get_x(self, double s)
    cpdef complex get_dxds(self, double s)
    cpdef complex[:] get_y(self, double s)
    cpdef complex[:] analytically_continue(self, complex xi, complex[:] yi, complex xip1)

cdef class RiemannSurfacePath(RiemannSurfacePathPrimitive):
    cdef int segment_index_at_parameter(self, double s)

cdef class RiemannSurfacePathPuiseux(RiemannSurfacePathPrimitive):
    cdef object puiseux_series
    cdef object target_point
    cdef object target_place

cdef class RiemannSurfacePathSmale(RiemannSurfacePathPrimitive):
    cdef int _degree
    cdef Wrapper_el[:] _df
