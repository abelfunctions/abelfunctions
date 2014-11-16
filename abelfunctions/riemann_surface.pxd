from .differentials cimport Differential
from .riemann_surface_path cimport RiemannSurfacePathPrimitive

cdef class RiemannSurface:
    cdef object _f
    cdef object _x
    cdef object _y
    cdef int _deg
    cdef object _base_point
    cdef object _base_sheets
    cdef object _discriminant_points
    cdef object _discriminant_points_exact
    cdef object _period_matrix
    cdef object _riemann_matrix
    cdef object _genus
    cdef object _holomorphic_differentials
    cdef object PathFactory
    cpdef complex integrate(self, Differential, RiemannSurfacePathPrimitive)
