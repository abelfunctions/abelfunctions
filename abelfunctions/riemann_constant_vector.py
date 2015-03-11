r"""Riemann Constant Vector :mod:`abelfunctions.riemann_constant_vector`
====================================================================

Module for computing the Riemann constant vector (RCV. Algorithm based
on the paper [DPS].

The computation of the RCV is based on observations from two
theorems. The the first theorem relates the RCV evaluated at a base
place, :math:`K(P_0)` to the Abel map evaluated at a canonical divisor:

.. math::

    2 K(P_0) \equiv - A(P_0,C).

The second theorem relates the RCV to the Riemann theta function: for
all effective, degree :math:`g-1` divisors :math:`D`,

.. math::

    \theta( K(P_0) + A(P_0,D), \Omega) = 0.

Classes
-------

RiemannConstantVector_Function

Functions
---------

canonical_divisor
compute_K0
find_regular_places
half_lattice_filter
half_lattice_vector
initialize_half_lattice_vectors


References
----------

.. [DPS] B. Deconinck, M.S. Patterson, and C. Swierczewski, "Computing
   the Riemann constant vector", (submitted for review, 2015)

Functions
---------

riemann_constant_vector

Examples
--------

Contents
--------
"""

from .abelmap import AbelMap, Jacobian
from .divisor import Place
from .riemanntheta import RiemannTheta

import numpy
from numpy import dot
from itertools import product

def initialize_half_lattice_vectors(X):
    r"""Generate a list of all half-lattice vectors.

    There are :math:`2^{2g}` half lattice vectors to consider. This
    returns a list of all of them. A generator is not necessary due to
    the filtration process. (We need to check every single half-lattice
    vector.)

    Parameters
    ----------
    X : RiemannSurface

    Returns
    -------
    list

    """
    g = X.genus()
    Omega = X.riemann_matrix()

    # compute a list of all vectors in {0,1/2}^g
    half = list(product((0,0.5),repeat=g))
    half_lattice_vectors = []

    # from each pair of vectors in {0,1/2}^g compute and store the
    # corresponding half-lattice vector
    for h1 in half:
        h1 = numpy.array(h1, dtype=numpy.complex).reshape((g,1))
        for h2 in half:
            h2 = numpy.array(h2, dtype=numpy.complex).reshape((g,1))
            h = h1 + dot(Omega,h2)
            half_lattice_vectors.append(h)
    return half_lattice_vectors

def half_lattice_filter(half_lattice_vectors, J, C, D, epsilon=1e-8):
    r"""Filter out any incorrect half-lattice vectors.

    This function tests if

    .. math::

        h + A(P_0,D) - \tfrac{1}{2}A(P_0,C)

    is a member of the theta divsior of the Riemann surface up to
    numerical accuracy `epsilon` for each vector :math:`h` appearing in
    `half_lattice_vectors`.

    Parameters
    ----------
    half_lattice_vectors : list
        A list of remaining half-lattice vectors to consider.
    J : Jacobian
    C : Divisor
        A canonical divisor of the surface.
    D : Divisor
        An effective, degree g-1 divisor.
    epsilon : double
        The tolerance to use when evaluating the Riemann theta
        function. Defaults to 1e-8.

    Returns
    -------
    list
        A filtered list of half-lattice vectors.

    """
    Z = AbelMap(D) - 0.5*AbelMap(C)

    # if we try a simple half_lattice_vectors.remove(h) then we get an
    # error "ValueError: The truth value of an array with more than one
    # element is ambiguous. Use a.any() or a.all()". Therefore, we need
    # to do an index tracking in conjunction with pop
    n = len(half_lattice_vectors)
    j = 0
    while j < n:
        h = half_lattice_vectors[j]
        kappa = J(h.T + Z)
        theta = RiemannTheta.oscillatory_part(kappa, J.Omega, prec=epsilon)
        if abs(theta) > epsilon:
            half_lattice_vectors.pop(j)
            n -= 1
            j -= 1
        j += 1
    return half_lattice_vectors

def find_regular_places(X, n):
    r"""Returns `n` regular places on `X`.

    This function begins at the x-origin and works "outward" along the
    real x-axis looking for points sufficiently far away from any
    discriminant points of the curve. At an appropriate point, the
    regular places lying above it are computed and added to a list.

    We choose integral x-points because it tends to simplify the Puiseux
    computations.

    Parameters
    ----------
    X : RiemannSurface
    n : integer

    Notes
    -----
    This should eventually move to a separate/different module, maybe.
    """
    # we use the X-path factory to find x-points that are bounded far
    # enough away from the discriminant points of the curve
    XPF = X.PF.XPF
    places = []

    a = 0
    while len(places) < n:
        b = X.closest_discriminant_point(a, exact=False)
        R = XPF.radius(b)

        # compute regular places if we are far enough away from any
        # discriminant points
        if abs(a-b) > R:
            places.extend(X(a))

        # pick a new x
        if x > 0: x = -x
        else:     x += 1

    # we obtain d = deg_y(f) places at a time. truncate to desired
    # number of places
    places = places[:n]
    return places


def sum_partitions(n):
    r"""A generator of all length n tuples :math:`(m_1,...,m_n)` such that

    .. math::

        m_1 + \cdots + m_n = n,

    where each :math:`m_i \geq 0`. Used by :func:`half_lattice_vector`
    to generate a bunch of effective degree g-1 divisors.

    Parameters
    ----------
    n : int

    Returns
    -------
    generator

    """
    # create the cartesian product of {0,...,n}
    cartesian = product(range(n+1), repeat=n)
    for p in cartesian:
        if sum(p) == n:
            yield p
    return


def half_lattice_vector(X, C, epsilon1, epsilon2):
    r"""Returns an appropriate half-lattice vector for the RCV.

    Parameters
    ----------
    X : RiemannSurface
    C : array
        A canonical divisor on the Riemann surface.
    """
    # create the list of all half-lattice vectors
    h = initialize_half_lattice_vectors(X)
    J = Jacobian(X)
    g = X.genus()

    # evaluate the Abel map at the canonical divisor
    if C.degree != (2*g-2):
        raise ValueError('C must be a canonical divisor.')

    # filter pass #1: D = (g-1)*P0
    D = (g-1)*X.base_place()
    h = half_lattice_filter(h, J, C, D, epsilon=epsilon1)
    if len(h) == 1:
        return h[0].T
    if len(h) == 0:
        raise AssertionError('Filtered out all half-lattice vectors.')

    # filter pass #2: D = sum of g-1 distinct regular places
    places = find_regular_places(X,g-1)
    D = sum(places)
    h = half_lattice_filter(h, J, C, D, epsilon=epsilon2)
    if len(h) == 1:
        return h[0].T
    if len(h) == 0:
        raise AssertionError('Filtered out all half-lattice vectors.')

    # filter pass #3: iterate over every degree g-1 divisor using the
    # places computed above
    for m in sum_partitions(g-1):
        D = reduce(lambda a,b: a[0]*a[1] + b[0]*b[1], zip(m,places))
        h = half_lattice_filter(h, J, C, D, epsilon=epsilon2)
        if len(h) == 1:
            return h[0].T
    if len(h) == 0:
        raise AssertionError('Filtered out all half-lattice vectors.')

    raise ValueError('Could not find appropriate lattice vector.')


def canonical_divisor(X):
    r"""Computes a canonical divisor on X.

    Selects a canonical divisor on X. Tries to select a place resulting
    in the easiest Abel map to compute by performing the following
    filters:

    * minimize on number of distinct places: (more places results in
      more paths)
    * prefer divisors containing only finite places: (paths to infinity
      need more testing)

    Parameters
    ----------
    X : RiemannSurface

    Returns
    -------
    Divisor

    Notes
    -----
    It takes time to compute the valuation divisors of the Abelian
    differentials of the first kind, in the first place. There may be a
    way to rewrite this algorithm so that it picks a "local best"
    canonical divisor.

    """
    holomorphic_oneforms = X.holomorphic_oneforms()
    canonical_divisors = [omega.valuation_divisor()
                          for omega in holomorphic_oneforms]

    # only take the divisors with the fewest number of distinct places
    N = min(len(C.places) for C in canonical_divisors)
    canonical_divisors = filter(lambda C: len(C.places)==N, canonical_divisors)

    # if there is a divisor with no infinite places, return that
    # one. otherwise, return any (the first) divisor computed
    for C in canonical_divisors:
        has_infinite_place = any(P.is_infinite() for P,n in C)
        if not has_infinite_place:
            return C
    return canonical_divisors[0]


def compute_K0(X, epsilon1, epsilon2):
    r"""Determine a base value of the Riemann Constant Vector.

    Given a Riemann surface `RS` and a canonical divisor `C` compute the
    Riemann Constant Vector at the base place.

    Parameters
    ----------
    RS : RiemannSurface
    C : Divisor

    Returns
    -------
    numpy.array
        The value of the RCV at the base place of the Riemann surface.

    """
    C = canonical_divisor(X)
    h = half_lattice_vector(X, C, epsilon1, epsilon2)
    J = Jacobian(X)
    K0 = J(h - 0.5*AbelMap(C))
    return K0


class RiemannConstantVector_Function(object):
    r"""The Riemann Constant Vector function.

    The Riemann Constant Vector (RCV) is used to parameterize the Theta
    divisor of a Riemann surface :math:`X`.

    Methods
    -------
    eval

    """
    def __init__(self, epsilon1=1e-6, epsilon2=None):
        r"""Initialize with numerical tolerances.

        We check for membership of the RCV in the Theta divisor by
        evaluating the Riemann theta function. A larger `epsilon1` is
        used in a first pass. A smaller `epsilon2` is used in a second
        pass in case the first pass does not produce a unique result.

        Parameters
        ----------
        epsilon1 : double
            Riemann theta tolerance used in the first pass. Set to 1e-6
            by default.
        epsilon2 : double
            Riemann theta tolerance used in all subsequent passes. Set
            to :math:`\epsilon_1 / 100` by default.

        """
        self.epsilon1 = epsilon1
        if epsilon2:
            self.epsilon2 = epsilon2
        else:
            self.epsilon2 = epsilon1*(1e-2)

        self._X = None
        self._J = None
        self._K0 = None

    def __call__(self, *args, **kwds):
        r"""Alias for :meth:`eval`."""
        return self.eval(*args, **kwds)

    def eval(self, P):
        r"""Evaluate the Riemann constant vector at the place `P`.

        Internally, the value of the RCV at the base place :math:`P_0`
        of the Riemann surface containing :math:`P` is computed and
        stored. This is done because the value of the RCV at any other
        place can be later computed using shift with the Abel map.

        Parameters
        ----------
        P : Place

        Returns
        -------
        array

        """
        if not isinstance(P, Place):
            raise ValueError('P must be a Place of a Rieamnn surface.')

        # recompute K(P0) if a place from a new Riemann surface is given
        if self._X != P.RS:
            self._X = P.RS
            self._J = Jacobian(self._X)
            self._K0 = compute_K0(self._X, self.epsilon1, self.epsilon2)

        # return K(P0) if P is the base place. otherwise, shift by the
        # abel map and return
        if P == self._X.base_place():
            return self._K0
        g = numpy.complex(self._X.genus()) # XXX necessary
        return self._J(self._K0 + (g-1)*AbelMap(P))

RiemannConstantVector = RiemannConstantVector_Function()
