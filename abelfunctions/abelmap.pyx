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

To invoke the Abel map in `abelfunctions` use `AbelMap(P)` where `P` is
a place on a Riemann surface or `AbelMap(D)` where `D` is a divisor on a
Riemann surface.


Classes
-------

AbelMap_Function

Examples
--------

Contents
--------
"""

import numpy
cimport numpy

from .divisor import Place, Divisor
from .riemann_surface cimport RiemannSurface
from .riemann_surface_path cimport RiemannSurfacePathPrimitive
from .differentials cimport Differential

cdef class AbelMap_Function:
    def __init__(self):
        pass

    def __call__(self, D):
        r"""Alias for :method:`eval`."""
        return self.eval(D)

    def eval(self, D):
        r"""Evaluate the Abel map at a place or divisor.

        Constructs a `RiemannSurfacePath` :math:`\gamma : [0,1] \to X`
        starting at the base place :math:`P_0` and ending at
        :math:`P`. Then, simply integrates each of the holomorphic
        one-forms along this path.

        Parameters
        ----------
        P : Place or Divisor
            The target place or divisor.

        Returns
        -------
        numpy.ndarray
            A length g array.

        """
        cdef int i,j,genus = D.RS.genus()
        cdef complex[:] val = numpy.zeros(genus,dtype=complex)
        cdef complex[:] Pval = numpy.zeros(genus,dtype=complex)
        cdef complex n
        if isinstance(D, Place):
            val = self._eval_primitive(D)
        else:
            for i in range(genus):
                n,P = D[i]
                Pval = self._eval_primitive(P)
                for j in range(genus):
                    val[j] = val[j] + n*Pval[j]
        return numpy.array(val,dtype=complex)

    cpdef complex[:] _eval_primitive(self, P):
        r"""Primitive evaluation of the Abel map at a single place, `P`.

        In the case when the input to :meth:`AbelMap_Function.eval` is a
        divisor the Abel map is computed at each place first.

        Parameters
        ----------
        P : Place

        Returns
        -------
        complex[:]
            A complex g-vector equal to the abel map
        """
        cdef RiemannSurface RS = P.RS
        cdef int i,genus = RS.genus()
        cdef complex[:] val = numpy.zeros(genus,dtype=complex)
        cdef RiemannSurfacePathPrimitive gamma = RS.path(P)
        cdef Differential[:] omega = numpy.array(
            RS.holomorphic_differentials(), dtype=Differential)
        for i in range(genus):
            val[i] = RS.integrate(omega[i],gamma)
        return val

AbelMap = AbelMap_Function()
