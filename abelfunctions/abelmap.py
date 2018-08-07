r"""Abel Map and Jacobian :mod:`abelfunctions.abelmap`
==================================================

Given a Riemann surface :math:`X` and a fixed place :math:`P_0 \in X` The Abel
Map is a function :math:`A : X \to \mathbb{C}^g` given by

.. math::

    A(P) := \left(
            \int_{P_0}^P \omega_1, \ldots, \int_{P_0}^P \omega_g
            \right)

where :math:`\omega_1, \ldots, \omega_g` are the normalized basis of
holomorphic one-forms defined on :math:`X`. The Abel map can be accessed via
``AbelMap()``, which is an instantiation of the class
:class:`AbelMap_Function`.

:class:`AbelMap_Function` makes use of :class:`Jacobian`, also defined
here. ``Jacobian`` defines the Jacboian of a Riemann surface

.. math::

    J(X) := \mathbb{C}^g / (\mathbb{Z}^g + \Omega \mathbb{Z}^g)

where :math:`\Omega` is the Riemann matrix of the Riemann surface. The primary
use of the Jacobian is to reduce vectors in :math:`\mathbb{C}^g` modulo this
lattice. The Jacobian is constructed with ``Jacobian(X)``.

Classes
-------

.. autosummary::

   AbelMap_Function
   Jacobian

Functions
---------

.. autosummary::

   fractional_part

Notes
-----

The implementation of :class:`AbelMap_Function` is based on the algorithm
described in [CTAM]_. When calling ``AbelMap()`` on a place :math:`P \in X` a
path on the Riemannn surface is constructed to it from the base place
:math:`P_0 \in X`. See
:meth:`abelfunctions.riemann_surface_path_factory.RiemannSurfacePathFactory.path_to_place`
for details on how this path is constructed.

Once this path is constructed, each of the non-normalized basis Abelian
differentials of the first kind computed using
:func:`abelfunctions.integralbasis.integral_basis` are integrated along
it. Multiplying the result by :math:`A^{-1}`, the inverse of the
:math:`a`-cycles periods, is equivalent to integrating the normalized
holomorphic differentials.

The implementation of :class:`Jacobian` uses the fact that

References
----------

.. [CTAM] B. Deconinck and M. S. Patterson, Computing the Abel map, Phys. D 237
   (2008), no. 24, 3214-3232. MR 2477016 (2010d:37139)

.. [CARS] A. I. Bobenko, Introduction to compact Riemann surfaces,
   Computational approach to Riemann surfaces, Lecture Notes in Math.,
   vol. 2013, Springer, Heidelberg, 2011, pp. 3-64. MR 2905610

Examples
--------

We construct a Riemann surface corresponding to the curve :math:`f(x,y) = x^2
y^3 - x^4 +1` and evaluate the Abel map at the single place :math:`P \in X`
lying above the point :math:`x=0`.

>>> import numpy; numpy.set_printoptions(precision=4, suppress=True)
>>> from sympy.abc import x,y
>>> from abelfunctions import RiemannSurface, AbelMap, Jacobian
>>> f = x**2*y**3 - x**4 + 1
>>> X = RiemannSurface(f,x,y)
>>> P = X(0)[0]
>>> AbelMap(P)
[-0.5261+0.0864j, 0.0669+0.6392j, -0.7495+1.1037j, -1.5030+1.0356j]

The Abel map can be computed between two different places. Let :math:`Q \in X`
be the sum of all places lying above the point :math:`x=2`. We compute
:math:`A(P,Q)` below.

>>> Q = sum(X(2))
>>> AbelMap(P,Q)
[ 0.1468-0.0985j,  0.8467+0.6989j,  0.0996+1.0083j, -1.1003+0.8159j]

Contents
--------
"""
import numpy

from sage.all import cached_method

def fractional_part(z, tol=1e-8):
    r"""Returns the fractional part of a vector.

    This function is different from ``numpy.floor``, which also determines the
    fractional part of a vector, in the sense that it can handle components
    that are *close* to integers. That is ``fractional_part(0.9999999999)`` and
    ``fractional_part(1.000000000001)`` should both return ``0`` since they're
    close to an integer.

    This is primarily used in :class:`Jacobian` to reduce a vector in
    :math:`\mathbb{C}^g` modulo the lattice :math:`\mathbb{Z}^g + \Omega
    \mathbb{Z}^g`.

    Parameters
    ----------
    z : double array
    tol=1e-8 : double
        Tolerance for determining when an entry is close to an integer.

    Returns
    -------
    w : array
    """
    # subtract off the integer part
    z = numpy.array(z)
    w = z - numpy.floor(z)

    # zero out any component of the form
    #
    #   w[i] = 1 - tol
    #
    # if any component is close to an integer, (in this case the integer should
    # be 1) set it equal to zero
    w[numpy.isclose(w,1)] = 0
    return w

class Jacobian(object):
    r"""The Jacobian of a Riemann Surface.

    The Jacboian of a Riemann surface is defined by

    .. math::

        J(X) := \mathbb{C}^g / (\mathbb{Z}^g + \Omega \mathbb{Z}^g)

    where :math:`\Omega` is the Riemann matrix of the Riemann surface. The
    primary use of the Jacobian is to reduce vectors in :math:`\mathbb{C}^g`
    modulo this lattice. The Jacobian is constructed with ``Jacobian(X)``.

    Attributes
    ----------
    Omega : complex matrix
        The Riemann matrix of the surface.
    g : int
        The genus of the Riemann surface
    M : complex matrix
        A transformation matrix used to decompose a vector :math:`z` into its
        components :math:`z = u + \Omega v`.

    Methods
    -------
    components
    eval

    """
    def __init__(self, X):
        r"""Initialize using a :class:`RiemannSurface`.

        Parameters
        ----------
        X : RiemannSurface
            A Riemann surface.

        """
        Omega = X.riemann_matrix()
        g = X.genus()

        M = numpy.zeros((2*g,2*g), dtype=numpy.double)
        M[:g,:g] = numpy.eye(g)  # upper left block = I
        M[:g,g:] = Omega.real    # upper right block = Re(Omega)
        M[g:,g:] = Omega.imag    # lower right block = Im(Omega)

        self.Omega = Omega
        self.M = M
        self.g = g

    def __call__(self, *args, **kwds):
        r"""Alias to :meth:`eval`."""
        return self.eval(*args, **kwds)

    def eval(self, z):
        r"""Reduce `z` modulo the lattice defining the Jacobian.

        Parameters
        ----------
        z : complex array
            An array / vector of g components.

        Returns
        -------
        zmod : complex array
            The vector z reduced modulo the lattice Lambda.
        """
        alpha, beta = self.components(z)
        alpha = fractional_part(alpha)
        beta = fractional_part(beta)
        zmod = alpha + numpy.dot(self.Omega, beta)
        return zmod

    def components(self, z):
        r"""Decomposes `z` into its lattice components :math:`z = u + \Omega v`.

        Parameters
        ----------
        z : complex array
            An array / vector of g components.

        Returns
        -------
        u, v : complex array
            Arrays / vectors of g components.

        Notes
        -----
        This is often used with :meth:`reduce_components`. In some cases when
        `z` is equal lattice element, floating point error can case
        :meth:`reduce_components` to incorrectly indicate this. (Two zero
        vectors should be returned in this case.) At the end of this routine we
        round the results to the nearest 15th decimal place in an attempt to
        rectify this error.

        """
        # reduces z = alpha + Omega*beta into its fractional components alpha
        # and beta
        g = self.g
        w = numpy.zeros(2*g, dtype=numpy.double)
        w[:g] = z.real[:]
        w[g:] = z.imag[:]

        # solve linear system to decompose z = z1 + Omega z2
        v = numpy.linalg.solve(self.M,w)

        # round to the nearest 15 digits due to possible floating point error
        # in the components. see note in description.
        z1 = v[:g]
        z2 = v[g:]
        return z1,z2


class AbelMap_Function(object):
    r"""The Abel Map.

    .. math::

        A(P) := \left(
                \int_{P_0}^P \omega_1, \ldots, \int_{P_0}^P \omega_g
                \right)

    By default, the Abel map is computed from the base place of the Riemann
    surface. Optionally, a starting place can be provided. That is,
    ``AbelMap(P,Q)`` returns

    .. math::

        A(P) := \left(
                \int_P^Q \omega_1, \ldots, \int_{P_0}^P \omega_g
                \right).

    The argument, :math:`Q`, can also be a divisor.

    Methods
    -------
    eval
    _eval_primitive

    """
    def __init__(self):
        pass

    def __call__(self, *args):
        r"""Alias for :method:`eval`."""
        return self.eval(*args)

    def eval(self, *args):
        r"""Evaluate the Abel map at a place or divisor.

        When only one argument (a place or divisor) ``D`` is provided, return
        :math:`A(P_0,D)`. If two arguments are given, a place ``P`` and a
        divisor ``D``, then return :math:`A(P,D)`.

        Constructs a `RiemannSurfacePath` :math:`\gamma : [0,1] \to X` starting
        at the base place :math:`P_0` and ending at :math:`P`. Then, simply
        integrates each of the holomorphic one-forms along this path.

        Parameters
        ----------
        P : Place, optional
            If two arguments are given, the first argument corresponds to the
            starting place of the Abel map. By default, this is the base place
            of the Riemann surface.
        D : Place or Divisor
            The target place or divisor of the Abel Map.

        Returns
        -------
        value : complex array
            The Abel map :math:`A(P,D)`.

        """
        if len(args) > 2:
            raise ValueError('Too many arguments.')
        elif len(args) == 2:
            # it's always assumed that the first input is a place and that the
            # inputs live on the same Riemann surface
            P,D = args
            if P.degree != 1:
                raise ValueError('First argument must be a place. (A divisor '
                                 'of order one.)')
            if P.RS != D.RS:
                raise ValueError('Inputs must be on the same Riemann surface')

            # perform the necessary transformation when the first place is
            # changed:
            #
            #   A(P,D) = A(P0,D) - (deg D)A(P0,P)
            #
            J = Jacobian(P.RS)
            value = self.eval(D) - (D.degree)*self.eval(P)
            return J(value)

        # compute the sum of the scaled Abel maps on the consituent places of
        # the divisor: if
        #
        #   D = \sum_i n_i P_i
        #
        # then
        #
        #   A(P0,D) = \sum n_i A(P0,P_i)
        #
        D = args[0]
        X = D.RS
        g = X.genus()
        value = numpy.zeros(g, dtype=numpy.complex)
        for P,n in D:
            Pvalue = self._eval_primitive(P)
            value += n*Pvalue

        # the definition of the Abel map involves the normalized holomorphic
        # differentials. achieve the same result by scaling the output with
        # respect to the "normalization" matrix A^{-1} of the period matrix
        #
        # TODO: use state so that the jacobian doesn't have to re recalculated
        # with every evaluation
        J = Jacobian(X)
        tau = X.period_matrix()
        A = tau[:g,:g]
        value = numpy.linalg.solve(A,value)
        return J(value)

    @cached_method
    def _eval_primitive(self, P):
        r"""Primitive evaluation of the Abel map at a single place, `P`.

        In the case when the input to :meth:`AbelMap_Function.eval` is a
        divisor the Abel map is computed at each place first. Always starts
        from the base place :math:`P_0` of the Riemann surface.

        Parameters
        ----------
        P : Place
            The target place.

        Returns
        -------
        value : complex array
            A complex g-vector equal to the Abel map evaluated at :math:`P`.

        """
        X = P.RS
        genus = X.genus()
        if P == X.base_place:
            value = numpy.zeros(genus, dtype=numpy.complex)
        else:
            gamma = X.path(P)
            omega = X.differentials
            value = numpy.array([X.integrate(omegai,gamma)
                                 for omegai in omega], dtype=numpy.complex)
        return value

AbelMap = AbelMap_Function()
