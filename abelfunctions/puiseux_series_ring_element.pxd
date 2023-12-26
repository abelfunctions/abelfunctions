from sage.structure.element cimport AlgebraElement, ModuleElement
from sage.rings.laurent_series_ring_element cimport LaurentSeries

cdef class PuiseuxSeries(AlgebraElement):
     cdef ModuleElement __l
     cdef long __e
