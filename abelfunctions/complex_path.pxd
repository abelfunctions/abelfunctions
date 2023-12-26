cdef class ComplexPathPrimitive:
    cdef ComplexPathPrimitive[:] _segments
    cdef int _nsegments
    cpdef complex eval(self, double s)
    cpdef complex derivative(self, double s)

cdef class ComplexPath(ComplexPathPrimitive):
    cdef int segment_index_at_parameter(self, double s)

cdef class ComplexLine(ComplexPathPrimitive):
    cdef complex _x0
    cdef complex _x1

cdef class ComplexArc(ComplexPathPrimitive):
    cdef double _R
    cdef complex _w
    cdef double _theta
    cdef double _dtheta

cdef class ComplexRay(ComplexPathPrimitive):
    cdef complex _x0

