cdef class ComplexPathPrimitive:
    cdef ComplexPathPrimitive[:] _segments
    cdef int _nsegments
    cpdef object eval(self, object s)
    cpdef object derivative(self, object s)

cdef class ComplexPath(ComplexPathPrimitive):
    cdef int segment_index_at_parameter(self, object s)

cdef class ComplexLine(ComplexPathPrimitive):
    cdef object _x0
    cdef object _x1

cdef class ComplexArc(ComplexPathPrimitive):
    cdef object _R
    cdef object _w
    cdef object _theta
    cdef object _dtheta

cdef class ComplexRay(ComplexPathPrimitive):
    cdef object _x0

