r"""Riemann Constant Vector :mod:`abelfunctions.riemann_constant_vector`
====================================================================

Module for computing the Riemann constant vector :math:`K : X \to J(X)`,

.. math::

    K(P) &= \left( K_1(P), \ldots, K_g(P) \right) \\
    K_j(P) &= \frac{1 + \Omega_{jj}}{2} -
             \sum_{k \neq j} \oint_{a_j} \omega_k(Q) A_j(P,Q) dQ

where :math:`A = (A_1, \ldots, A_g)` is the Abel map. (See
:class:`abelfunctions.abelmap.AbelMap_Function`.) The Riemann constant vector
is an essential ingredient to computing finite genus solutions to integrable
systems as well as characterizing the theta divisor of a Riemann surface.

The algorithm for computing the RCV is based on the one described in [DPS]_. It
relies on observations from two theorems. The the first theorem relates the RCV
evaluated at a base place, :math:`K(P_0)` to the Abel map evaluated at a
canonical divisor:

.. math::

    2 K(P_0) \equiv - A(P_0,C).

The second theorem relates the RCV to the Riemann theta function: for all
effective, degree :math:`g-1` divisors :math:`D`,

.. math::

    \theta( K(P_0) + A(P_0,D), \Omega) = 0.

Functions
---------

.. autosummary::

    canonical_divisor
    compute_K0
    find_regular_places
    half_lattice_filter
    half_lattice_vector
    initialize_half_lattice_vectors
    RiemannConstantVector


References
----------

.. [DPS] B. Deconinck, M.S. Patterson, and C. Swierczewski, "Computing
   the Riemann constant vector", (submitted for review, 2015)

Examples
--------

We evaluate the Riemann constant vector at the base place of the genus four
Riemann surface corresponding to the plane algebraic curve :math:`f(x,y) =
x^2y^3 - x^4 + 1`.

>>> from sympy.abc import x,y
>>> from abelfunctions import RiemannSurface, RiemannConstantVector
>>> f = x**2*y**3 - x**4 + 1
>>> X = RiemannSurface(f,x,y)
>>> P0 = X.base_place()
>>> RiemannConstantVector(P0)
[ 0.8488+0.7203j -0.5941-0.1146j -0.7432+0.8913j -0.8189+1.1381j]

Contents
--------
"""

from abelfunctions.abelmap import AbelMap, Jacobian
from abelfunctions.divisor import Place
from abelfunctions.riemann_theta import RiemannTheta

import numpy
from numpy import dot
from itertools import product

from sage.all import cached_function

def initialize_half_lattice_vectors(X):
    r"""Generate a list of all half-lattice vectors.

    There are :math:`2^{2g}` half lattice vectors to consider. This returns a
    list of all of them. A generator is not necessary due to the filtration
    process. (We need to check every single half-lattice vector.)

    Parameters
    ----------
    X : RiemannSurface

    Returns
    -------
    half_lattice_vectors : list

    """
    g = X.genus()
    Omega = X.riemann_matrix()

    # compute a list of all vectors in {0,1/2}^g
    half = list(product((0,0.5),repeat=g))
    half_lattice_vectors = numpy.array(
        [h1 + dot(Omega,h2) for h1 in half for h2 in half],
        dtype=numpy.complex
    )
    return half_lattice_vectors


def half_lattice_filter(half_lattice_vectors, J, C, D, epsilon=1e-8):
    r"""Filter out any incorrect half-lattice vectors.

    This function tests if

    .. math::

        h + A(P_0,D) - \tfrac{1}{2}A(P_0,C)

    is a member of the theta divsior of the Riemann surface up to numerical
    accuracy `epsilon` for each vector :math:`h` appearing in
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
    half_lattice_vectors : list
        A filtered list of half-lattice vectors.

    """
    # construct the set of "shifted" half-lattice vectors: the vectors J(h +
    # A(D) - 0.5*A(C)) where h is a half-lattice vector
    Z = AbelMap(D) - 0.5*AbelMap(C)
    shifted_half_lattice_vectors = [J(elt) for elt in half_lattice_vectors + Z]

    # evaluate Riemann theta at each of these half-lattice vectors.
    theta_values = RiemannTheta.oscillatory_part(
        shifted_half_lattice_vectors, J.Omega, epsilon=epsilon
    )
    theta_values = abs(theta_values)

    # return only the half-lattice vectors for which the corresponding theta
    # values are less than epsilon in absolute value
    half_lattice_vectors = half_lattice_vectors[theta_values < epsilon]
    return half_lattice_vectors


def find_regular_places(X, n):
    r"""Returns `n` regular places on `X`.

    This function begins at the x-origin and works "outward" along the real
    x-axis looking for points sufficiently far away from any discriminant
    points of the curve. At an appropriate point, the regular places lying
    above it are computed and added to a list.

    We choose integral x-points because it tends to simplify the Puiseux
    computations.

    Parameters
    ----------
    X : RiemannSurface
    n : integer

    Returns
    -------
    places : list
    """
    # we use the X-path factory to find x-points that are bounded far enough
    # away from the discriminant points of the curve
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

        # pick a new a
        if a > 0:
            a = -a
        else:
            a += 1

    # we obtain deg_y(f) places at a time. truncate to desired number of places
    places = places[:n]
    return places


def sum_partitions(n):
    r"""A generator of all length n tuples :math:`(m_1,...,m_n)` such that

    .. math::

        m_1 + \cdots + m_n = n,

    where each :math:`m_i \geq 0`. Used by :func:`half_lattice_vector` to
    generate a collection of effective degree g-1 divisors.

    Parameters
    ----------
    n : int

    Returns
    -------
    p : generator

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
    C : Divisor
        A canonical divisor on the Riemann surface.

    Returns
    -------
    h : array
    """
    # create the list of all half-lattice vectors
    h = initialize_half_lattice_vectors(X)
    J = Jacobian(X)
    g = X.genus()

    # filter pass #1: D = (g-1)*P0
    D = (g-1)*X.base_place
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

    # filter pass #3: iterate over every degree g-1 divisor using the places
    # computed above
    for m in sum_partitions(g - 1):
        D = sum(a * b for a, b in zip(m, places))
        h = half_lattice_filter(h, J, C, D, epsilon=epsilon2)
        if len(h) == 1:
            return h[0].T
    if len(h) == 0:
        raise AssertionError('Filtered out all half-lattice vectors.')

    raise ValueError('Could not find appropriate lattice vector.')


@cached_function
def canonical_divisor(X):
    r"""Computes a canonical divisor on X.

    Selects a canonical divisor on X. Tries to select a place resulting in the
    easiest Abel map to compute by performing the following filters:

    * minimize on number of distinct places: (more places results in more
      paths)
    * prefer divisors containing only finite places: (paths to infinity need
      more testing)

    Parameters
    ----------
    X : RiemannSurface

    Returns
    -------
    C : Divisor

    Notes
    -----
    It takes time to compute the valuation divisors of the Abelian
    differentials of the first kind, in the first place. There may be a way to
    rewrite this algorithm so that it picks a "local best" canonical divisor.

    """
    holomorphic_oneforms = X.differentials
    canonical_divisors = [omega.valuation_divisor()
                          for omega in holomorphic_oneforms]

    # only take the divisors with the fewest number of distinct places
    N = min(len(C.places) for C in canonical_divisors)
    canonical_divisors = [C for C in canonical_divisors if len(C.places) == N]

    # if there is a divisor with no infinite places, return that
    # one. otherwise, return any (the first) divisor computed
    for C in canonical_divisors:
        has_infinite_place = any(P.is_infinite() for P, n in C)
        if not has_infinite_place:
            return C
    return canonical_divisors[0]


@cached_function
def compute_K0(X, epsilon1, epsilon2, C):
    r"""Determine a base value of the Riemann Constant Vector.

    Given a Riemann surface `RS` and a canonical divisor `C` compute the
    Riemann Constant Vector at the base place.

    Parameters
    ----------
    X : RiemannSurface
    epsilon1, epsilon2 : double
        Numerical accuracy thresholds. See :func:`half_lattice_filter`.
    C : Divisor
        A canonical divisor on the Riemann surface.

    Returns
    -------
    K0 : array
        The value of the RCV at the base place of the Riemann surface.

    """
    h = half_lattice_vector(X, C, epsilon1, epsilon2)
    J = Jacobian(X)
    K0 = J(h - 0.5*AbelMap(C))
    return K0


def RiemannConstantVector(P, epsilon1=1e-6, epsilon2=1e-8, C=None):
    r"""Evaluate the Riemann constant vector at the place `P`.

    Internally, the value of the RCV at the base place :math:`P_0` of the
    Riemann surface containing :math:`P` is computed and stored. This is done
    because the value of the RCV at any other place can be later computed using
    shift with the Abel map.

    Parameters
    ----------
    P : Place
    epsilon1 : double, optional
        Riemann theta tolerance used in the first pass. Default: ``1e-6``.
    epsilon2 : double, optional
        Tolerance used in all subsequent passes. Default: ``1e-8``.
    C : Divisor, optional
        A canonical divisor on the Riemann surface. Computes one using
        :func:`canonical_divisor` if no such divisor is provided.

    Returns
    -------
    K : array
        The Riemann constant vector at `P`.

    """
    if not isinstance(P, Place):
        raise ValueError('P must be a Place of a Rieamnn surface.')

    # check if C is a canonical divisor if one is provided
    if C is not None:
        degree = C.degree
        n = numpy.array(C.multiplicities)
        if (degree != (2*P.RS.genus()-2)) or any(n < 0):
            raise ValueError("Cannot compute Riemann constant vector: given "
                             "divisor %s is not canonical."%C)
        if C.RS != P.RS:
            raise ValueError("Cannot compute Riemann constant vector: the "
                             "place %s and the canonical divisor %s do not "
                             "live on the same Riemann surface."%(P,C))
    else:
        C = canonical_divisor(P.RS)

    # return K0 =K(P0) if P is the base place. otherwise, perform appropriate
    # shift by the abel map
    X = P.RS
    J = Jacobian(X)
    g = numpy.complex(X.genus())
    K0 = compute_K0(X, epsilon1, epsilon2, C)
    if P == X.base_place:
        return K0
    return J(K0 + (g-1)*AbelMap(P))
