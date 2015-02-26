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

import numpy

class Jacobian(object):
    def __init__(self, X):
        Omega = X.riemann_matrix()
        g = X.genus()

        M = numpy.zeros((2*g,2*g), dtype=numpy.double)
        M[:g,:g] = numpy.eye(g)  # upper left block = I
        M[:g,g:] = Omega.real    # upper right block = Re(Omega)
        M[g:,g:] = Omega.imag    # lower right block = Im(Omega)

        self.Omega = Omega
        self.M = M
        self.g = g

    def __call__(self, z):
        r"""Reduce `z` modulo the lattice defining the Jacobian.

        Parameters
        ----------
        z : complex array
            An array / vector of g components.

        Returns
        -------
        z mod Lambda
            The vector z reduced modulo the lattice Lambda.
        """
        alpha, beta = self.reduced_components(z)
        zmod = alpha + numpy.dot(self.Omega, beta)
        return zmod.T

    def components(self, z):
        """Decomposes `z` into its components :math:`z=z_1+\Omega z_2`.

        Parameters
        ----------
        z : complex array
            An array / vector of g components.

        Returns
        -------
        z1, z2
            Arrays / vectors of g components.

        Notes
        -----
        This is often used with :meth:`reduce_components`. In some cases
        when `z` is equal lattice element, floating point error can case
        :meth:`reduce_components` to incorrectly indicate this. (Two
        zero vectors should be returned in this case.) At the end of
        this routine we round the results to the nearest 15th decimal
        place in an attempt to rectify this error.

        """
        # reduces z = alpha + Omega*beta into its fractional components
        # alpha and beta
        g = self.g
        z = z.reshape((g,1))
        w = numpy.zeros((2*g,1), dtype=numpy.double)
        w[:g] = z.real[:]
        w[g:] = z.imag[:]

        # solve linear system to decompose z = z1 + Omega z2
        v = numpy.linalg.solve(self.M,w)

        # round to the nearest 15 digits due to possible floating point
        # error in the components. see note in description.
        z1 = numpy.around(v[:g], decimals=15)
        z2 = numpy.around(v[g:], decimals=15)
        return z1,z2

    def reduced_components(self, z):
        r"""Decomposes `z` into its components :math:`z=\alpha+\Omega\beta`
        where :math:`\alpha,\beta \in [0,1)^g`.

        Parameters
        ----------
        z : complex array
            An array / vector of g components.

        Returns
        -------
        alpha, beta
            Arrays / vectors of g components.

        """
        z1, z2 = self.components(z)
        alpha = z1 - numpy.floor(z1)
        beta = z2 - numpy.floor(z2)
        return alpha,beta


cdef class AbelMap_Function:
    def __init__(self):
        pass

    def __call__(self, *args):
        r"""Alias for :method:`eval`."""
        return self.eval(*args)

    def eval(self, *args):
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

        .. note::

            TODO: Use state so that the Jacobian doesn't have to be
            recalculated every time AbelMap is evaluated.

        """
        if len(args) == 2:
            D1,D2 = args
            X = D1.RS
            J = Jacobian(X)
            return J(self.eval(D2) - self.eval(D1))
        elif len(args) != 1:
            raise ValueError('Too many arguments.')

        D = args[0]

        cdef RiemannSurface RS = D.RS
        cdef int i,j
        cdef int genus = RS.genus()
        cdef complex[:] val = numpy.zeros(genus,dtype=complex)
        cdef complex[:] Pval = numpy.zeros(genus,dtype=complex)
        cdef complex n
        if isinstance(D, Place):
            val = self._eval_primitive(D)
        else:
            for P,n in D:
                Pval = self._eval_primitive(P)
                for j in range(genus):
                    val[j] = val[j] + n*Pval[j]

        # TODO: use state so that the jacobian doesn't have to re
        # recalculated with every evaluation
        J = Jacobian(RS)
        return J(numpy.array(val,dtype=complex))

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
