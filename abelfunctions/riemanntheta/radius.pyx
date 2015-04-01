r"""Riemann Theta Precision Radius :mod:`abelfunctions.riemanntheta.radius`
=======================================================================

Functions for computing the primary radius of the bounding ellipsoid of the
oscillatory part of the Riemann theta function.

Each subroutine solves for the radius using:

* Theorem 3 of [CRTF] (no derivatives)
* Theorem 5 of [CRTF] (first order derivative)
* Theorem 7 of [CRTF] (second order derivative)
* extrapolation of theorems (higher order derivatives, not implemented)

Functions
---------

.. autosummary::

    radius
    raidus0
    raidus1
    radius2

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

from numpy import sqrt, prod
from numpy.linalg import norm, inv
from scipy.special import gamma, gammaincc, gammainccinv
from scipy.optimize import fsolve

cdef extern from 'lll_reduce.h':
    void lll_reduce(double*, int, double, double);

def radius(self, epsilon, double[:,:] T, derivs=[], accuracy_radius=5):
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
    T : double[:,:]
        Cholesky decomposition of imaginary part of the Riemann matrix.
    derivs : double[:,:]
        A list of lists representing directional derivatives of the
        Riemann theta function.
    accuracy_radius : double
        Radius for guaranteed region of accuracy. See above.
    """
    # compute the LLL-reduction of T
    cdef double[:,:] A = numpy.array(T, dtype=numpy.double)
    g = numpy.double(A.shape[0])
    lll_reduce(&A[0,0], g, .50, .75)
    A = numpy.array(A, dtype=numpy.double)
    r = min(norm(A[:,i]) for i in range(int(g)))

    if len(derivs) == 0:
        radius = radius0(epsilon, r, g)
    elif len(derivs) == 1:
        radius = radius1(epsilon, r, g, T, derivs[0], accuracy_radius)
    elif len(derivs) == 2:
        radius = radius2(epsilon, r, g, T, derivs, accuracy_radius)
    else:
        raise NotImplementedError('Cannot yet compute higher derivatives of '
                                  'the Riemann theta function.')
    return radius


def radius0(eps, r, g):
    r"""Compute the radius with no deriviatives."""
    lhs = eps * (2./g) * (r/2.)**g * gamma(g/2.)
    ins = gammainccinv(g/2.,lhs)
    R = sqrt(ins) + r/2.
    S = (sqrt(2.*g)+r)/2.
    radius = max(R,S)
    return radius


def radius1(eps, r, g, T, deriv, accuracy_radius=5):
    r"""Compute the radius with one deriviative."""
    pi = numpy.pi
    L = accuracy_radius
    normderiv = norm(numpy.array(deriv))
    normTinv = norm(inv(T))
    lhs = (eps*(r/2.)**g) / (sqrt(pi)*g*normderiv*normTinv)

    # define lower bound (guess) and attempt to solve for the radius
    lbnd = sqrt(g+2+sqrt(g**2+8)) + r
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


def radius2(eps, r, g, T, derivs, accuracy_radius):
    r"""Compute the radius with two deriviatives."""
    pi = numpy.pi
    L = accuracy_radius
    prodnormderiv = prod([numpy.array(d) for d in derivs])
    normTinv = norm(inv(T))
    lhs = (eps*(r/2.0)**g) / (2*pi*g*prodnormderiv*normTinv**2)

    # define lower bound (guess) and attempt to solve for the radius
    lbnd = sqrt(g+4+sqrt(g**2+16)) + r
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

