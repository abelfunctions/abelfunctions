cdef class RiemannSurface:
    cdef object _f
    cdef object _x
    cdef object _y
    cdef int _deg
    cdef object _period_matrix
    cdef object PathFactory

