r"""
Differentials :mod:`abelfunctions.differentials`
================================================

This module contains functions for computing a basis of holomorphic
differentials of a Riemann surface given by a complex plane algebraic
curve :math:`f \in \mathbb{C}[x,y]`. A differential :math:`\omega =
h(x,y)dx` defined on a Riemann surface :math:`X` is holomorphic on
:math:`X` if it is holomorphic at every point on :math:`X`.

The function :func:`differentials` computes the basis of holomorphic
differentials from an input algebraic curve :math:`f = f(x,y)`. The
differentials themselves are encapsulated in a :class:`Differential`
Cython class.

Classes
-------

.. autosummary::

    Differential

Functions
---------

.. autosummary::

    differentials
    mnuk_conditions

References
----------

.. [Mnuk] M. Mnuk, "An algebraic approach to computing adjoint curves",
   Journal of Symbolic Computation, vol. 23 (2-3), pp. 229–40, 1997.

Examples
--------

Contents
--------

"""
import numpy
import sympy
import sympy.mpmath as mpmath
import matplotlib
import matplotlib.pyplot as plt

from .integralbasis import integral_basis
from .singularities import singularities, _transform, genus
from .utilities import cached_function
from .polynomials cimport MultivariatePolynomial

cimport cython

def mnuk_conditions(g, u, v, b, P, c):
    """Determine the Mnuk conditions on the coefficients of :math:`P`.

    Determine the conditions on the coefficients `c` of `P` at the
    integral basis element `b` modulo the curve `g = g(u,v)`. See [Mnuk]
    for details.

    Parameters
    ----------
    g : sympy.Expr
    u : sympy.Symbol
    v : sympy.Symbol
    b : sympy.Expr
        An integral basis element.
    P : sympy.Expr
        A generic adjoint polynomial as provided by
        :func:`differentials`. Only one instance is created for caching
        and performance purposes.
    c : list, sympy.Symbol
        A list of the unknown symbolic coefficients we wish to solve
        for.

    Returns
    -------
    list, sympy.Expr
        A list of expressions from which a system of equations is build
        to determine the differentials.

    """
    numer, denom = b.as_numer_denom()

    # reduce b*P modulo g
    expr = numer.as_poly(v,u,*c) * P.as_poly(v,u,*c, domain='QQ[I]')
    q,r = sympy.reduced(expr,[sympy.poly(g,v,u,*c)])

    # divide by the largest power of x appearing in the denominator.
    # this is sufficient since we've shifted the curve and its
    # singularity to appear at
    try:
        mult = sympy.roots(denom.as_poly(u))[sympy.S(0)]
    except KeyError:
        mult = 0

    r = r.as_poly(u,v)
    coeffs = r.coeffs()
    monoms = r.monoms()
    conditions = [coeff for coeff,monom in zip(coeffs,monoms)
                  if monom[0] < mult]
    return conditions

def differentials(f, x, y):
    """Returns a basis of holomorphic differentials on Riemann surface.

    The surface is given by the desingularization and compactification
    of the affine complex plane algebraic curve `f = f(x,y)`.

    Parameters
    ----------
    f : sympy.Expr
    x : sympy.Symbol
    y : sympy.Symbol

    Returns
    -------
    list, Differential

    """
    # compute the "total degree" (Poly.total_degree doesn't give the
    # desired result). This is the largest monomial degree in the sum
    # of the degrees in both x and y.
    d = max(map(sum,f.as_poly(x,y).monoms()))
    n = sympy.degree(f,y)

    # define the "generalized" adjoint polynomial.
    c = sympy.symarray('c',(d-2,d-2)).tolist()
    P = sum( c[i][j] * x**i * y**j
             for i in range(d-2) for j in range(d-2)
             if i+j <= d-3)
    c = [cij for ci in c for cij in ci]

    # for each singular point [x:y:z] = [alpha:beta:gamma], map f onto
    # the "most convenient and appropriate" affine subspace, (u,v),
    # and center at u=0. determine the conditions on P
    S = singularities(f,x,y)
    conditions = []
    for singular_pt,(m,delta,r) in S:
        # recenter the curve and adjoint polynomial at the
        # singular point: find the affine plane u,v such that
        # the singularity occurs at u=0
        g,u,v,u0,v0 = _transform(f,x,y,singular_pt)
        g = g.subs(u,u+u0)
        Ptilde,u,v,u0,v0 = _transform(P,x,y,singular_pt)
        Ptilde = Ptilde.subs(u,u+u0)

        # compute the intergral basis at the recentered singular point
        # and determine the Mnuk conditions of the adjoint polynomial
        b = integral_basis(g,u,v)
        for bi in b:
            conditions_bi = mnuk_conditions(g,u,v,bi,Ptilde,c)
            conditions.extend(conditions_bi)

    # solve the system of equations and retreive the coefficents of the c_ij's
    # contained in the general solution
    sols = sympy.solve(conditions, c)
    P = P.subs(sols).as_poly(*c)
    differentials = [coeff for coeff in P.coeffs() if coeff != 0]

    # # sanity check: the number of differentials matches the genus
    # g = genus(f,x,y)
    # if g != -1 and g != len(differentials):
    #     raise AssertionError("Number of differentials does not match genus.")
    dfdy = sympy.diff(f,y)
    differentials = [differential/dfdy for differential in differentials]
    return map(lambda omega: Differential(omega, x, y), differentials)


cdef class Differential:
    """A differential one-form which can be defined on a Riemann surface.

    Attributes
    ----------
    numer, denom : MultivariatePolynomial
        Fast multivariate polynomial objects representing the numerator
        and denominator of the differential.

    Methods
    -------
    eval(z1,z2)
        Fast evaluation of the differential.
    as_sympy()
        Returns the differential as a Sympy object.

    """
    def __cinit__(self, omega, x, y):
        """Instantiate a differential form from a sympy Expression.

        Parameters
        ----------
        omega : Sympy Expression
        x, y : Sympy Symbol
            The differential and its variables. Note in abelfunctions we
            consider `y` to be a function of `x`. (A degree d y-cover.)

        """
        numer, denom = omega.as_numer_denom()
        numer = numer.expand()
        denom = denom.expand()
        self.numer = MultivariatePolynomial(numer, x, y)
        self.denom = MultivariatePolynomial(denom, x, y)
        self._omega = omega

    def __repr__(self):
        return str(self._omega)

    cpdef complex eval(self, complex z1, complex z2):
        """Evaluate the differential at the complex point :math:`(z_1,z_2)`.

        Parameters
        ----------
        z1,z2 : complex

        Returns
        -------
        complex
            Returns the value :math:`\omega(z_1,z_2)`.

        """
        return self.numer.eval(z1,z2) / self.denom.eval(z1,z2)

    def plot(self, gamma, N=256, grid=False, **kwds):
        """Plot the differential along the RiemannSurfacePath `gamma`.

        Parameters
        ----------
        gamma : RiemannSurfacePath
            A path along which to evaluate the differential.
        N : int
            Number of interpolating points to use when plotting the
            value of the differential along the path `gamma`
        grid : boolean
            (Default: `False`) If true, draw gridlines at each "segment"
            of the parameterized RiemannSurfacePath. See the
            `RiemannSurfacePath` documentation for more information.

        Returns
        -------
        matplotlib.Figure

        """
        nsegs = len(gamma.segments)
        ppseg = N/nsegs

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        t = numpy.linspace(0,1,ppseg)
        for k in range(nsegs):
            segment = gamma.segments[k]
            xvals = [segment.get_x(ti) for ti in t]
            yvals = [segment.get_y(ti)[0] for ti in t]
            ovals = numpy.array(
                [self.eval(xi,yi) for xi,yi in zip(xvals,yvals)],
                dtype=numpy.complex)

            tseg = (t + k)/nsegs
            ax.plot(tseg, ovals.real, 'b-', **kwds)
            ax.plot(tseg, ovals.imag, 'r--', **kwds)

        if grid:
            ticks = numpy.linspace(0,1,nsegs+1)
            ax.xaxis.set_ticks(ticks)
            ax.grid(True, which='major')

        return fig

    def as_sympy(self):
        """Returns the differential as a Sympy expression."""
        return self._omega

