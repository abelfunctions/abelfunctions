r"""Riemann Constant Vector :mod:`abelfunctions.riemann_constant_vector`
====================================================================

Module for computing the Riemann constant vector.

Functions
---------

riemann_constant_vector

Examples
--------

Contents
--------
"""

from .abelmap import AbelMap, Jacobian
from .riemanntheta import RiemannTheta

from numpy.linalg import dot
from itertools import product

def initalize_half_lattice_vectors(X):
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

    .. math:

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
    for h in half_lattice_vectors:
        kappa = J(h + Z)
        theta = RiemannTheta.oscillatory_part(kappa, J.Omega, prec=epsilon)
        if abs(theta) > epsilon:
            half_lattice_vectors.remove(h)
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
    cartesian = itertools.product(range(n+1), repeat=n)
    for p in cartesian:
        if sum(p) == n:
            yield p
    return


def half_lattice_vector(X, AbelC, epsilon, epsilon2):
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
    if C.degree() != (2*g-2):
        raise ValueError('C must be a canonical divisor.')
    AbelC = AbelMap(C)

    # filter pass #1: D = (g-1)*P0
    D = (g-1)*X.base_place()
    h = half_lattice_filter(h, J, AbelC, D, epsilon=epsilon)
    if len(h) == 1:
        return h[0]

    # filter pass #2: D = sum of g-1 distinct regular places
    places = find_regular_places(X,g-1)
    D = sum(places)
    h = half_lattice_filter(h, J, AbelC, D, epsilon=epsilon2)
    if len(h) == 1:
        return h[0]

    # filter pass #3: iterate over every degree g-1 divisor using the
    # places computed above
    for m in sum_partitions(g-1):
        D = reduce(lambda a,b: a[0]*a[1] + b[0]*b[1], zip(m,places))
        h = half_lattice_filter(h, J, AbelC, D, epsilon=epsilon2)
        if len(h) == 1:
            return h[0]

    raise ValueError('Could not find appropriate lattice vector.')


def canonical_divsior(X):
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
    N = min(lambda C: len(C.places), canonical_divisors)
    canonical_divisors = filter(lambda C: len(C.places)==N, canonical_divisors)

    # if there is a divisor with no infinite places, return that
    # one. otherwise, return any (the first) divisor computed
    for C in canonical_divisors:
        has_infinte_place = any(P.is_infinite() for P,n in C)
        if not has_infinite_place:
            return C
    return canonical_divisors[0]


def compute_K0(X, epsilon=1e-8, epsilon2=None):
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
    if epsilon2 is None:
        epsilon2 = epsilon*(1e-2)

    C = canonical_divisor(X)
    h = half_lattice_vector(X, C, epsilon, epsilon2)
    J = Jacobian(X)
    K0 = J(h - 0.5*AbelMap(C))
    return K0


class RiemannConstantVector(object):
    def __init__(self, X, epsilon=1e-8, epsilon2=None):
        self.K0 = compute_K0(X, epsilon=epsilon, epsilon2=epsilon2)
        self.g = X.genus()

    def __call__(self, P):
        if P == X.base_place():
            return self.K0
        return self.K0 + (self.g-1)*AbelMap(P)
