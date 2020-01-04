r"""Integral Basis :mod:`abelfunctions.integralbasis`
=================================================

A module for computing integral bases of algebraic function fields of the form
:math:`O(X) = \mathbb{C}[x,y] / (f(x,y))` where :math:`X : f(x,y) = 0`.

For polynomials over :math:`\mathbb{Q}[x,y]` we use Singular's very fast
implementation.

The slow / general-purpose algorithm is based off of the paper "An Algorithm
for Computing an Integral Basis in an Algebraic Function Field" by Mark van
Hoeij [vHoeij]_.

An integral basis for :math:`O(X)` is a set of :math:`\beta_i \in
\mathbb{C}(x,y)` such that

.. math::

    \overline{O(X)} = \beta_1\mathbb{C}[x,y] + \cdots + \beta_g\mathbb{C}[x,y].

This data is necessary for computing a basis for the space of holomorphic
differentials :math:`\Omega_X^1` defined on the Riemann surface :math:`X` which
is implemented in ``differentials``.

Functions
---------

.. autosummary::

    integral_basis

References
----------

.. [vHoeij] Mark van Hoeij. "An Algorithm for Computing an Integral Basis in an
   Algebraic Function Field". J. Symbolic Computation. (1994) 18, p. 353-363

.. [Singular] Wolfram Decker, Gert-Martin Greuel, Gerhard Pfister, and Hans
   Schonemann. "Singular: library for computing the normalization of affine
   rings". (2015).

Examples
--------

Contents
--------

"""

from abelfunctions.puiseux import puiseux
from abelfunctions.puiseux_series_ring import PuiseuxSeriesRing

from sage.all import cached_function
from sage.functions.other import ceil
from sage.matrix.constructor import Matrix
from sage.rings.polynomial.all import PolynomialRing
from sage.rings.rational_field import QQ
from sage.rings.qqbar import QQbar

import warnings

def Int(i, px):
    r"""Computes :math:`Int_i = \sum_{k \neq i} v(p_i-p_k)`.

    ``Int`` is used in :func:`compute_expansion_bounds` for determining
    sufficient bounds on Puiseux series expansions.

    Parameters
    ----------
    i : int
        Index of the Puiseux series in the list, `px`, to compute `Int` of.
    p : list, PuiseuxXSeries
        A list of :class:`PuiseuxXSeries`.

    Returns
    -------
    val : rational
       The `Int` of the `i`th element of `px`.

    """
    n = len(px)
    pxi = px[i]
    val = QQ(0)
    for k in range(n):
        if k != i:
            val += (pxi-px[k]).valuation()
    return val

def compute_expansion_bounds(px):
    r"""Returns a list of necessary bounds on each Puiseux series in ``px``.

    Computes the expansion bounds :math:`N_1, \ldots, N_n` such that for
    all polynomials :math:`G \in L[x,y]` the truncation :math:`r_i` of
    the Puiseux series :math:`p_i` satisfying :math:`v(r_i - p_i) > N_i`
    satisfies the relation

    .. math::

        \forall M,i, v(G(r_i)) > M

    if and only if

    .. math::

        \forall M,i, v(G(p_i)) > M.

    That is, the truncations :math:`r_i` are sufficiently long so that
    polynomial evaluation of :math:`r_i` and :math:`p_i` has the same
    valuation.

    Parameters
    ----------
    px : list, PuiseuxXSeries

    Returns
    -------
    list, int
        A list of degree bounds for each PuiseuxXSeries in ``px``.

    """
    n = len(px)
    N = []
    max_Int = max([Int(k, px) for k in range(n)])
    for i in range(n):
        pairwise_diffs = [(px[k] - px[i]).valuation()
                          for k in range(n) if k != i]
        Ni = max(pairwise_diffs) + max_Int - Int(i, px) + 1
        N.append(Ni)
    return N


def compute_series_truncations(f, alpha):
    r"""Computes Puiseux series at :math:`x=\alpha` with necessary terms.

    The Puiseux series expansions of :math:`f = f(x,y)` centered at
    :math:`\alpha` are computed up to the number of terms needed for the
    integral basis algorithm to be successful. The expansion degree bounds are
    determined by :func:`compute_expansion_bounds`.

    Parameters
    ----------
    f : polynomial
    alpha : complex

    Returns
    -------
    list : PuiseuxXSeries
        A list of Puiseux series expansions centered at :math:`x = \alpha` with
        enough terms to compute integral bases as SymPy expressions.

    """
    # compute the parametric Puiseix series with the minimal number of terms
    # needed to distinguish them.
    pt = puiseux(f,alpha)
    px = [p for P in pt for p in P.xseries()]

    # compute the orders necessary for the integral basis algorithm. the orders
    # are on the Puiseux x-series (non-parametric) so scale by the ramification
    # index of each series
    N = compute_expansion_bounds(px)
    for i in range(len(N)):
        e = px[i].ramification_index
        N[i] = ceil(N[i]*e)

    order = max(N) + 1
    for pti in pt:
        pti.extend(order=order)

    # recompute the corresponding x-series with the extened terms
    px = [p for P in pt for p in P.xseries()]
    return px


def integral_basis(f):
    r"""Returns the integral basis of the algebraic function field of `f`.

    An integral basis for the algebraic function field :math:`O(X)` is a
    set of :math:`\beta_i \in \mathbb{C}(x,y)` such that

    .. math::

        \overline{O(X)} = \beta_1 \mathbb{C}[x,y] + \cdots + \beta_g
        \mathbb{C}[x,y].

    Parameters
    ----------
    f : sympy.Expr
    x : sympy.Symbol
    y : sympy.Symbol

    Returns
    -------
    list, sympy.Expr
        A list of rational functions representing an integral basis.

    """
    R = f.parent()
    x,y = R.gens()

    # The base algorithm assumes f is monic. If this is not the case then
    # monicize by applying the map `y -> y/lc(x), f -> lc^(d-1) f` where lc(x)
    # is the leading coefficient of f.
    d  = f.degree(y)
    lc = f.polynomial(y).leading_coefficient()
    if lc.degree() > 0:
        # we have to carefully manage rings here. the path is:
        #     R(x)[y] -> R[x][y] -> R[x,y]
        fmonic = f(x,y/lc)*lc**(d-1) # element of R(x)[y]
        B = R.base_ring()
        fmonic = fmonic.change_ring(B[x])  # element of R[x][y]
        fmonic = R(fmonic)  # element of R[x,y]
    else:
        fmonic = f/R.base_ring()(lc)
        lc = 1

    # if the curve lives in QQ[x,y] then use singular. otherwise, use slow
    # self-implemented version
    try:
        fmonic = fmonic.change_ring(QQ)
    except:
        warnings.warn('using slower integral basis algorithm: '
                      'cannot coerce curve %s to QQ[%s,%s]'%(fmonic,x,y))
        b = _integral_basis_monic(fmonic)
    else:
        b = _integral_basis_monic_singular(fmonic)

    # reverse leading coefficient scaling
    for i in range(1,len(b)):
        b[i] = b[i](x,lc*y)
    return b


def _integral_basis_monic_singular(f):
    r"""Computes an integral basis using singular.

    Note that singular can only compute integral bases of algebraic function
    fields over :math:`\mathbb{Q}[x,y]`. It will fail over other extensions.

    Parameters
    ----------
    f : curve

    Returns
    -------
    b : list
        A list of integral basis elements.
    """
    from sage.all import singular
    singular.load('integralbasis.lib')

    l = singular.integralBasis(f,2)
    ideal, denom = l.sage()
    numerators = ideal.gens()
    b = [numer/denom for numer in numerators]
    return b


def _integral_basis_monic(f):
    r"""Returns the integral basis of a monic curve.

    Called by :func:`integral_basis` after monicizing its input curve.

    Parameters
    ----------
    f : polynomial

    Returns
    -------
    list : rational functions
        A list of rational functions representing an integral basis of the
        monic curve.

    See Also
    --------
    integral_basis : generic integral basis function

    """
    R = f.parent()
    x,y = R.gens()

    # compute df: the set of monic, irreducible polynomials k such that k**2
    # divides the resultant
    n = f.degree(y)
    res = f.resultant(f.derivative(y),y).univariate_polynomial()
    factor = res.squarefree_decomposition()
    df = [k for k,deg in factor
          if (deg > 1) and (k.leading_coefficient() == 1)]

    # for each element k of df, take any root of k and compute the
    # corresponding Puisuex series centered at that point
    r = []
    alpha = []
    for k in df:
        alphak = k.roots(ring=QQbar, multiplicities=False)[0]
        alpha.append(alphak)
        rk = compute_series_truncations(f,alphak)
        r.append(rk)

    # main loop
    b = [R.fraction_field()(1)]
    for d in range(1,n):
        bd = compute_bd(f,b,df,r,alpha)
        b.append(bd)
    return b

def compute_bd(f, b, df, r, alpha):
    """Determine the next integral basis element form those already computed."""
    # obtain the ring of Puiseux series in which the truncated series
    # live. these should already be such that the base ring is SR, the symbolic
    # ring. (below we will have to introduce symbolic indeterminants)
    R = f.parent()
    F = R.fraction_field()
    x,y = R.gens()

    # construct a list of indeterminants and a guess for the next integral
    # basis element. to make computations uniform in the univariate and
    # multivariate cases an additional generator of the underlying polynomial
    # ring is introduced.
    d = len(b)
    Q = PolynomialRing(QQbar, ['a%d'%n for n in range(d)] + ['dummy'])
    a = tuple(Q.gens())
    b = tuple(b)
    P = PuiseuxSeriesRing(Q, str(x))
    xx = P.gen()
    bd = F(y*b[-1])

    # XXX HACK
    for l in range(len(r)):
        for k in range(len(r[l])):
            r[l][k] = r[l][k].change_ring(Q)

    # sufficiently singularize the current integral basis element guess at each
    # of the singular points of df
    for l in range(len(df)):
        k = df[l]  # factor
        # alphak = alpha[l]  # point at which the truncated series are centered
        rk = r[l]  # truncated puiseux series

        # singularize the current guess at the current point using each
        # truncated Puiseux seriesx
        sufficiently_singular = False
        while not sufficiently_singular:
            # from each puiseux series, rki, centered at alphak construct a
            # system of equations from the negative exponent terms appearing in
            # the expression A(x,rki))
            equations = []
            for rki in rk:
                #                A = sum(a[j] * b[j](xx,rki) for j in range(d))
                A = evaluate_A(a,b,xx,rki,d)
                A += bd(xx, rki)

                # implicit division by x-alphak, hence truncation to x^1
                terms = A.truncate(1).coefficients()
                equations.extend(terms)

            # attempt to solve this linear system of equations. if a (unique)
            # solution exists then the integral basis element is not singular
            # enough at alphak
            sols = solve_coefficient_system(Q, equations, a)
            if not sols is None:
                bdm1 = sum(F(sols[i][0])*b[i] for i in range(d))
                bd = F(bdm1 + bd)/ F(k)
            else:
                sufficiently_singular = True
    return bd

@cached_function
def evaluate_A(a,b,xx,rki,d):
    A = sum(a[j] * b[j](xx,rki) for j in range(d))
    return A

def solve_coefficient_system(Q, equations, vars):
    # NOTE: to make things easier (and uniform) in the univariate case a dummy
    # variable is added to the polynomial ring. See compute_bd()
    a = Q.gens()[:-1]
    B = Q.base_ring()

    # construct the coefficient system and right-hand side
    system = [[e.coefficient({ai:1}) for ai in a] for e in equations]
    rhs = [-e.constant_coefficient() for e in equations]
    system = Matrix(B, system)
    rhs = Matrix(B, rhs).transpose()

    # we only allow unique solutions. return None if there are infinitely many
    # solutions or if no solution exists. Sage will raise a ValueError in both
    # circumstances
    try:
        sol = system.solve_right(rhs)
    except ValueError:
        return None
    return sol

