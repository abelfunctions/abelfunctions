"""
Analytic Continutation
======================

Objects for performing analytic continuation along a
`RiemannSurfacePath` object.


Authors
-------

* Chris Swierczewski (January 2013)
"""

import numpy
cimport numpy
import sympy

from riemann_surface cimport RiemannSurface
from riemann_surface_path cimport RiemannSurfacePathPrimitive

cdef extern from 'complex.h':
    double creal(complex)
    double cimag(complex)
    double cabs(complex)


cdef class AnalyticContinuator:
    """Abstract class for analytically continuing along a curve.

    Attributes
    ----------
    RS : RiemannSurface
        The Riemann surface on which analytic continuation takes place

    Methods
    -------
    analytically_continue(
            RiemannSurfacePathPrimitive gamma,
            complex xi,
            complex xip1,
            complex[:] yi)
        Analytically continues the fibre `yi` from `xi` to `xip1`.
    """
    def __init__(self,RiemannSurface RS):
        self.RS = RS
        self.deg = self.RS.deg

    cpdef complex[:] analytically_continue(
            self,
            RiemannSurfacePathPrimitive gamma,
            complex xi,
            complex[:] yi,
            complex xip1):
        """Analytically continues the fibre `yi` from `xi` to `xip1`.

        Arguments
        ---------
        gamma : RiemannSurfacePathPrimitive
            A Riemann surface path-type object.
        xi : complex
            The starting complex x-value.
        yi: complex[:]
            The starting complex y-fibre lying above `xi`.
        xip1: complex
            The target complex x-value.

        Returns
        -------
        complex[:]
            The y-fibre lying above `xip1`.
        """
        raise NotImplementedError('Must override AnalyticContinuator.'
                                  'analytically_continue() method in '
                                  'subclass.')
