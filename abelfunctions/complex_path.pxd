cdef class ComplexPathPrimitive:
    cdef ComplexPathPrimitive[:] _segments
    cdef int _nsegments
    cpdef object eval(self, double s)
    cpdef object derivative(self, double s)

cdef class ComplexPath(ComplexPathPrimitive):
    cdef int segment_index_at_parameter(self, double s)

cdef class ComplexLine(ComplexPathPrimitive):
    cdef object _x0
    cdef object _x1

cdef class ComplexArc(ComplexPathPrimitive):
    cdef double _R
    cdef object _w
    cdef double _theta
    cdef double _dtheta

cdef class ComplexRay(ComplexPathPrimitive):
    cdef object _x0

