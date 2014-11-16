r"""Differentials :mod:`abelfunctions.differentials`
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
from .riemann_surface_path cimport RiemannSurfacePathPrimitive

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
    numer,denom = b.as_numer_denom()

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
    # desired result). This is the largest monomial degree in the sum of
    # the degrees in both x and y.
    d = max(map(sum,f.as_poly(x,y).monoms()))
    n = sympy.degree(f,y)

    # define the "generalized" adjoint polynomial.
    c = sympy.symarray('c',(d-2,d-2)).tolist()
    P = sum( c[i][j] * x**i * y**j
             for i in range(d-2) for j in range(d-2)
             if i+j <= d-3)
    c = [cij for ci in c for cij in ci]

    # for each singular point [x:y:z] = [alpha:beta:gamma], map f onto
    # the "most convenient and appropriate" affine subspace, (u,v), and
    # center at u=0. determine the conditions on P
    S = singularities(f,x,y)
    conditions = []
    for singular_pt,(m,delta,r) in S:
        # recenter the curve and adjoint polynomial at the singular
        # point: find the affine plane u,v such that the singularity
        # occurs at u=0
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

    # solve the system of equations and retreive the coefficents of the
    # c_ij's contained in the general solution
    sols = sympy.solve(conditions, c)
    P = P.subs(sols).as_poly(*c)
    differentials = [coeff for coeff in P.coeffs() if coeff != 0]
    dfdy = sympy.diff(f,y)
    differentials = [differential/dfdy for differential in differentials]
    return map(lambda omega: Differential(omega, x, y), differentials)


def fast_expand(numer,denom,t,order):
    r"""Quickly compute the Taylor expansion of `numer/denom`.

    Parameters
    ----------
    numer, denom : sympy.Expr
        Polynomials in t.
    t : sympy.Symbol
        The dependent variable of `numer` and `denom`.
    order : int
        The desired order of the expansion.

    Returns
    -------
    sympy.Expr
    """
    # convert numerator and denominator to lists of coefficients.  it's
    # faster to do it "manually" than to coerce to polynomial
    q = [0]*order
    for term in numer.args:
        qn,n = term.as_coeff_exponent(t)
        if n < order:
            q[n] = qn
    r = [0]*order
    for term in denom.args:
        rn,n = term.as_coeff_exponent(t)
        if n < order:
            r[n] = rn

    # forward solve the coefficient system. note that r[0] (constant
    # coeff of denom) is nonzero by construction
    s = [0]*order
    for n in range(order):
        known_terms = sum(r[n-k]*s[k] for k in range(n))
        s[n] = (q[n] - known_terms)/r[0]
    taylor = sum(s[n]*t**n for n in range(order))
    return taylor

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
    def __cinit__(self,omega,x,y):
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
        self.x = x
        self.y = y
        self.numer = MultivariatePolynomial(numer, x, y)
        self.denom = MultivariatePolynomial(denom, x, y)
        self._omega = omega

    def __repr__(self):
        return str(self._omega)

    cpdef complex eval(self, complex x, complex y):
        r"""Evaluate the differential at the complex point :math:`(x,y)`.

        Parameters
        ----------
        x,y : complex

        Returns
        -------
        complex
            Returns the value :math:`\omega(x,y)`.

        """
        return self.numer.eval(x,y) / self.denom.eval(x,y)

    def centered_at_place(self, P):
        r"""Rewrite the differential in terms of the local coordinates at `P`.

        If `P` is a regular place, then returns `self` as a sympy
        expression. Otherwise, if `P` is a discriminant place
        :math:`P(t) = \{x(t), y(t)\}` then returns

        .. math::

            \omega |_P = q(x(t),y(t)) x'(t) / \partial_y f(x(t),y(t)).

        Parameters
        ----------
        P : Place

        Returns
        -------
        sympy.Expr
        """
        x = self.x
        y = self.y
        if P.is_discriminant():
            # evaluate the numerator and denominator separately up to
            # the order of the Puiseux series
            p = P.puiseux_series
            t = p.t
            xt = p.eval_x(t)
            yt = p.eval_y(t)
            dxdt = p.eval_dxdt(t)

            numer,denom = self.as_sympy_expr().as_numer_denom()
            numer = sympy.expand(numer.subs({x:xt,y:yt})*dxdt)
            denom = sympy.expand(denom.subs({x:xt,y:yt}))
            numer,denom = sympy.cancel(numer/denom).as_numer_denom()
            omega = fast_expand(numer,denom,t,p.order)
        else:
            omega = self._omega
        return omega

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex[:] evaluate(self, RiemannSurfacePathPrimitive gamma,
                              double[:] t):
        r"""Evaluates `omega` along the path at `N` uniform points.

        .. todo::

            Note: right now it doesn't matter what the values in `t`
            are. This function will simply turn `t` into a bunch of
            uniformly distributed points between 0 and 1.

        Parameters
        ----------
        omega : Differential
        t : double[:]
            An array of `t` between 0 and 1.

        Returns
        -------
        complex[:]
            The differential omega evaluated along the path at `N` points.
        """
        return gamma.evaluate(self,t)


    def plot(self, RiemannSurfacePathPrimitive gamma, N=256, grid=False,
             **kwds):
        r"""Plot the differential along the RiemannSurfacePath `gamma`.

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
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hold(True)

        nseg = len(gamma.segments)
        t = numpy.linspace(0,1,N/nseg)
        for k in range(nseg):
            segment = gamma.segments[k]
            osegment = numpy.array(self.evaluate(segment,t),dtype=complex)
            tsegment = (t+k)/nseg;
            ax.plot(tsegment, osegment.real, 'b')
            ax.plot(tsegment, osegment.imag, 'r--')

        # plot gridlines at the interface between each set of segments
        if grid:
            ticks = numpy.linspace(0,1,len(gamma.segments)+1)
            ax.xaxis.set_ticks(ticks)
            ax.grid(True, which='major')
        return fig


    def as_sympy_expr(self):
        """Returns the differential as a Sympy expression."""
        return self._omega

