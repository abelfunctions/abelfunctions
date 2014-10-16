r"""Abel Map :mod:`abelfunction.abelmap`
====================================

The Abel Map is a function :math:`A : X \to \mathbb{C}^g` given by

.. math::

    A(P) = \left(
           \int_{P_0}^P \omega_1, \ldots, \int_{P_0}^P \omega_g
           \right)

where

* :math:`X` is a genus g Riemann surface,

* :math:`P_0` is a fixed "base place" on the Riemann surface,

* :math:`\omega_1, \ldots, \omega_g` are a basis of g holomorphic
  one-forms defined on the surface,

* :math:`P` is a target place

Classes
-------

AbelMap_Function

Examples
--------

Contents
--------
"""

from .riemann_surface cimport RiemannSurface
from .riemann_surface_path cimport RiemannSurfacePathPrimitive
from .differentials cimport HolomorpicDifferential

class AbelMap_Function(object):
    def __init__(self, RiemannSurface RS):
        self.RS = RS

    def __call__(self, P):
        r"""Alias for :method:`eval`."""
        return self.eval(P)

    cpdef complex[:] eval(self, P):
        r"""Evaluate the Abel map at a place, `P`.

        Constructs a `RiemannSurfacePath` :math:`\gamma : [0,1] \to X`
        starting at the base place :math:`P_0` and ending at
        :math:`P`. Then, simply integrates each of the holomorphic
        one-forms along this path.

        Parameters
        ----------
        P : Place
            The target place.

        Returns
        -------
        numpy.ndarray
            A length g array.

        """
        cdef RiemannSurfacePathPrimitive gamma = self.RS.path(P)
        cdef Differential[:] omegas = self.RS.holomorphic_oneforms()
        cdef complex[:] val = numpy.array(
            [self.RS.integrate(omega, gamma) for omega in omegas],
            dtype=numpy.complex)
        return val
