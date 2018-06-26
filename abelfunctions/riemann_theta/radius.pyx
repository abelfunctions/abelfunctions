r"""Riemann Theta Precision Radius :mod:`abelfunctions.riemanntheta.radius`
=======================================================================

Functions for computing the primary radius of the bounding ellipsoid of the
oscillatory part of the Riemann theta function.

The two subroutines solve for the radius using,

* Theorem 3 of [CRTF] (no derivatives)
* A generalization of Theorems 5 and 7 of [CRTF] for N derivatives

Functions
---------

.. autosummary::

    radius
    radius0
    radius1
    radius2
    radiusN

References
----------

.. [CRTF] B. Deconinck, M.  Heil, A. Bobenko, M. van Hoeij and M. Schmies,
   Computing Riemann Theta Functions, Mathematics of Computation, 73, (2004),
   1417-1442.

.. [DLMF] B. Deconinck, Digital Library of Mathematical Functions - Riemann
   Theta Functions, http://dlmf.nist.gov/21

Contents
--------

"""
import numpy
import scipy
import warnings

from numpy import sqrt, prod
from numpy.linalg import norm, inv
from scipy.special import gamma, gammaincc, gammainccinv, binom
from scipy.optimize import fsolve

cdef extern from *:
    void lll_reduce(double*, int, double, double)


def lll(M, lc=.5, uc=.75):
    r"""Returns the LLL-reduction of the columns of `M`.

    Parameters
    ----------
    M : array
        A real, square matrix.
    lc, uc : double
        Parameters between 0 and 1 used in the LLL algorithm.

    Returns
    -------
    array
    """
    cdef int g = M.shape[0]
    cdef double dlc=lc, duc=uc
    cdef double[:,:] A = numpy.ascontiguousarray(
        M.astype(numpy.double), dtype=numpy.double)
    lll_reduce(&A[0,0], g, dlc, duc)
    return numpy.array(A, dtype=numpy.double)


def radius(epsilon, T, derivs=[], accuracy_radius=5):
    r"""Returns the primary radius of the bounding ellipsoid for computing the
    Riemann theta function up to accuracy `epsilon`.

    The derivative oscillatory part of the Riemann theta function has linear
    growth in :math:`z` along the directions of the columns of the Riemann
    matrix. `accuracy_radius` is used to determine a radius resulting in an
    accurate Riemann theta for all

    .. math ::

        ||z|| < \text{accuracy_radius}.

        Parameters
    ----------
    epsilon : double
        Requested accuracy.
    T : matrix
        A gxg matrix representing the Cholesky decomposition of the imaginary
        part of a Riemann matrix.
    derivs : list of lists
        (Default: []) A list of directional derivatives. The number of
        directional derivatives is the order, N, of the derivative we wish to
        compute.
    accuracy_radius : double
        (Default: 5) Radius for guaranteed region of accuracy. See above.

    Returns
    -------
    radius : double
        The initial radius of the bounding ellipsoid used to truncate the
        Riemann theta function to desired accuracy.

    """
    # compute the LLL-reduction of sqrt(pi)*T
    T = numpy.array(T, dtype=numpy.double)
    g = T.shape[0]
    U = lll(sqrt(numpy.pi)*T)
    r = min(norm(U[:,i]) for i in range(g))

    if len(derivs) == 0:
        radius = radius0(epsilon, r, g)
    elif len(derivs) > 0:
        radius = radiusN(epsilon, r, g, T, derivs,
                         accuracy_radius=accuracy_radius)
    else:
        raise TypeError('Expected list of lists representing '
                'directional derivative.')

    return radius


def radius0(eps, r, g):
    r"""Compute the radius with no derivatives.

    Parameters
    ----------
    eps : double
        Requested accuracy.
    r : double
        The length of the shortest lattice vector in the LLL reduction of the
        Cholesky decomposition of the imaginary part of the Riemann matrix
        times sqrt(pi).
    g : int
        The genus / problem size.

    Returns
    -------
    radius : double
        The initial radius of the bounding ellipsoid used to truncate the
        Riemann theta function to desired accuracy.

    """
    lhs = eps * (2./g) * (r/2.)**g / gamma(g/2.)
    ins = gammainccinv(g/2.,lhs)
    R = sqrt(ins) + r/2.
    S = (sqrt(2.*g)+r)/2.
    radius = max(R,S)
    return radius


def radiusN(eps, r, g, T, derivs, accuracy_radius=5):
    r"""Compute the radius with N derivatives.

    Parameters
    ----------
    eps : double
        Requested accuracy.
    r : double
        The length of the shortest lattice vector in the LLL reduction of the
        Cholesky decomposition of the imaginary part of the Riemann matrix
        times sqrt(pi).
    g : int
        The genus / problem size.
    T : matrix
        A gxg matrix representing the Cholesky decomposition of the imaginary
        part of a Riemann matrix.
    derivs : list of lists
        A list of directional derivatives. The number of directional
        derivatives is the order, N, of the derivative we wish to compute.
    accuracy_radius : double
        Radius for guaranteed region of accuracy. See :func:`radius`.

    Returns
    -------
    radius : double
        The initial radius of the bounding ellipsoid used to truncate the
        Riemann theta function to desired accuracy.

    """
    N = len(derivs)
    pi = numpy.pi
    L = accuracy_radius
    prodnormderiv = prod([norm(d) for d in derivs])
    normTinv = norm(inv(T))
    lhs = (eps*r**g*2**(1-g-N)) / (pi**(N/2.)*g*normTinv**N*prodnormderiv)

    # define lower bound (guess) and attempt to solve for the radius
    lbnd = (sqrt(g + 2*N + sqrt(g**2 + 8*N)) + r)/2.
    def rhs(ins):
        A = [binom(N,k) * pi**(k/2.) * (L*normTinv)**k * gamma((g+N-k)/2.) * \
             gammaincc((g+N-k)/2.,ins) for k in range(N+1)]
        A = sum(A)
        B = lhs
        return A - B

    try:
        ins = fsolve(rhs, lbnd)[0]
    except RuntimeWarning:
        # try a larger initial guess. worse case scenario we have better
        # Riemann theta precision
        try:
            ins = fsolve(rhs, 2*lbnd)[0]
        except RuntimeWarning:
            raise ValueError('Could not compute Riemann theta finite sum '
                             'bounding ellipsoid. Try using better precision.')

    R = sqrt(ins) + r/2.0
    radius = max(R,lbnd)
    return radius


def radius1(eps, r, g, T, deriv, accuracy_radius=5):
    r"""Compute the radius with one derivative.

    Notes
    -----
    Depreciated. Use `radiusN` instead. `radius1` is only used for testing
    purposes.
    """

    warnings.warn('radius1 is only for testing purposes. Use `radiusN` '
                  'instead.', DeprecationWarning)

    pi = numpy.pi
    L = accuracy_radius
    normderiv = norm(numpy.array(deriv))
    normTinv = norm(inv(T))
    lhs = (eps*(r/2.)**g) / (sqrt(pi)*g*normderiv*normTinv)

    # define lower bound (guess) and attempt to solve for the radius
    lbnd = (sqrt(g+2+sqrt(g**2+8)) + r)/2.
    def rhs(ins):
        A = gamma((g+1)/2.) * gammaincc((g+1)/2., ins)
        B = sqrt(pi) * normTinv * L * gamma(g/2.) * gammaincc(g/2., ins)
        C = lhs
        return A + B - C

    try:
        ins = fsolve(rhs, lbnd)[0]
    except RuntimeWarning:
        # try a larger initial guess. worse case scenario we have better Riemann
        # theta precision
        try:
            ins = fsolve(rhs, 2*lbnd)[0]
        except RuntimeWarning:
            raise ValueError('Could not compute Riemann theta finite sum '
                             'bounding ellipsoid. Try using better precision.')

    R = sqrt(ins) + r/2.0
    radius = max(R,lbnd)
    return radius


def radius2(eps, r, g, T, derivs, accuracy_radius=5):
    r"""Compute the radius with two derivatives.

    Notes
    -----
    Depreciated. Use `radiusN` instead. `radius2` is only used for testing
    purposes.
    """

    warnings.warn('radius2 is only for testing purposes. Use `radiusN` '
                  'instead.', DeprecationWarning)

    pi = numpy.pi
    L = accuracy_radius
    prodnormderiv = prod([norm(d) for d in derivs])
    normTinv = norm(inv(T))
    lhs = (eps*(r/2.0)**g) / (2*pi*g*prodnormderiv*normTinv**2)

    # define lower bound (guess) and attempt to solve for the radius
    lbnd = (sqrt(g+4+sqrt(g**2+16)) + r)/2.
    def rhs(ins):
        A = gamma((g+2)/2.) * gammaincc((g+2)/2.,ins)
        B = 2*L*sqrt(pi) * normTinv * gamma((g+1)/2.) * gammaincc((g+1)/2.,ins)
        C = pi * normTinv**2 * L**2 * gamma(g/2.) * gammaincc(g/2.,ins)
        D = lhs
        return A + B + C - D

    try:
        ins = fsolve(rhs, lbnd)[0]
    except RuntimeWarning:
        # try a larger initial guess. worse case scenario we have better Riemann
        # theta precision
        try:
            ins = fsolve(rhs, 2*lbnd)[0]
        except RuntimeWarning:
            raise ValueError('Could not compute Riemann theta finite sum '
                             'bounding ellipsoid. Try using better precision.')

    R = sqrt(ins) + r/2.0
    radius = max(R,lbnd)
    return radius

