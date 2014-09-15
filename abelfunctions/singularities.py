r"""Singularities :mod:`abelfunctions.singularities`
================================================

A module for computing the singular points of a complex plane algebraic
curve including their multiplicities, branching numbers, multiplicities,
and delta invariants.

Each singularity :math:`P = (x,y,z)` on the projectivization of the
curve has associated with it a three-tuple :math:`(m, \delta, r)`
representing the multiplicity, delta invariant, and branching number of
the multiplicity.

Functions
---------

singularities

Examples
--------

Contents
--------
"""
import sympy

import pdb

from abelfunctions.puiseux import puiseux
from abelfunctions.integralbasis import Int

def homogenize(f, x, y, z):
    r"""Homogenizes the curve :math:`f = f(x,y)`.

    Parameters
    ----------
    f, x, y : sympy.Expr, sympy.Symbol
        The curve in its original affine coordinates.
    z : sympy.Symbol
        The additional projective coordinate variable.

    Returns
    -------
    sympy.Expr
        The projective curve :math:`F(z,x,y) = z^d f(x/z,y/z)`.

    """
    p = sympy.poly(f,[x,y])
    d = max(map(sum,p.monoms()))
    F = sympy.expand(z**d*f.subs([(x,x/z),(y,y/z)]))
    return F,d


def _singular_points_finite(f, x, y):
    r"""Returns the finite singular points of `f`.

    Parameters
    ----------
    f, x, y : sympy.Expr, sympy.Symbol
        An affine algebraic curve.

    Returns
    -------
    list
        A list of three-tuples :math:`(x_k,y_k,1)` representing the
        affine (finite) singular points of the curve :math:`f = f(x,y)`

    """
    S = []

    # compute the finite singularities: use the resultant to find the
    # x-points at which singularities may occur
    p  = sympy.Poly(f,[x,y])
    n  = p.degree(y)
    res = sympy.Poly(sympy.resultant(p,p.diff(y),y),x)
    for xk,deg in sympy.roots(res,x).iteritems():
        if deg > 1:
            # for each y-root ykj above xk, record a singular point if
            # the gradient vanishes at (xk,ykj)
            fxk = sympy.Poly(f.subs({x:xk}),y)
            for ykj,_ in sympy.roots(fxk,y).iteritems():
                fx = f.diff(x)
                fy = f.diff(y)
                subs = {x:xk,y:ykj}
                if (fx.subs(subs) == 0) and (fy.subs(subs) == 0):
                    S.append((xk,ykj,1))
    return S


def _singular_points_infinite(f, x, y):
    r"""Returns the singular points of `f` at infinity.

    In particualr, returns all of the projective singular points
    :math:`(x_k,y_k,z_k)` such that :math:`z_k = 0`.

    Parameters
    ----------
    f, x, y : sympy.Expr, sympy.Symbol
        An affine algebraic curve.

    Returns
    -------
    list
        A list of three-tuples :math:`(x_k,y_k,0)` representing the
        infinite singular points of the curve :math:`f = f(x,y)`

    """
    # compute homogenous polynomial
    _z = sympy.Symbol('z')
    F,d = homogenize(f,x,y,_z)

    # find the possible singular points at infinity:
    domain = sympy.QQ[sympy.I]
    F0 = F.subs({_z:0})
    F0X1 = sympy.Poly(F0.subs({x:1,y:_z}),_z,domain=domain)
    F0Y1 = sympy.Poly(F0.subs({x:_z,y:1}),_z,domain=domain)
    solsX1 = sympy.roots(F0X1).keys()
    solsY1 = sympy.roots(F0Y1).keys()
    all_points = [(1,yi,0) for yi in solsX1]
    all_points.extend([(xi,1,0) for xi in solsY1])

    # these possible singularities are in projective space, so filter
    # out equal points such as (0,1,I) == (0,-I,1). normalize these
    # projective points such that 1 appears in either the x- or y-
    # coordinate, where appropriate
    normalized_points = []
    for xi,yi,zi in all_points:
        P = (1,yi/xi,0) if xi != sympy.S(0) else (xi/yi,1,0)
        if not P in normalized_points:
            normalized_points.append(P)

    # finally, check the gradient condition to get the actual singular points
    S = []
    grad = [F.diff(var) for var in (x,y,_z)]
    for xi,yi,zi in normalized_points:
        fsub = lambda e,x=x,y=y,_z=_z,xi=xi,yi=yi,zi=zi:  \
               e.subs({x:xi,y:yi,_z:zi}) != sympy.S(0)
        if not any(map(fsub,grad)):
            S.append((xi,yi,zi))

    return S


def singularities(f, x, y):
    r"""Returns the singularities of the curve `f` in projective space.

    Returns all of the projective singular points :math:`(z_k,x_k,y_k)`
    on the curve :math:`f(x,y) = 0`. For each singular point a
    three-tuple :math:`(m, \delta, r)` is given representing the
    multiplicity, delta invariant, and branching number of the
    singularity.

    This information is used to resolve singularities for the purposes
    of computing a Riemann surface. The singularities are resolved using
    Puiseux series.

    Parameters
    ----------
    f : sympy.Expr
    x : sympy.Symbol
    y : sympy.Symbol

    Returns
    -------
    list
        A list of the singularities, both finite and infinite, along
        with their multiplicity, delta invariant, and branching number
        information.

    """
    S = _singular_points_finite(f,x,y)
    S_oo = _singular_points_infinite(f,x,y)
    S.extend(S_oo)

    info = []
    for singular_pt in S:
        # Perform a projective transformation of the curve so it's
        # almost centered at the singular point.
        g,u,v,u0,v0 = _transform(f, x, y, singular_pt)
        P = puiseux(g, u, v, u0, v0, parametric=True)

        m = _multiplicity(P)
        delta = _delta_invariant(P)
        r = _branching_number(P)

        info.append((m,delta,r))

    return zip(S,info)


def _transform(f, x, y, singular_pt):
    r"""Recenters the affine curve `f` at a singular point.

    Returns :math:`(g,u,v,u0,v0)` where :math:`g = g(u,v)` is the
    transformed polynomial and :math:`u0,v0` is the projection of
    :math:`[\alpha : \beta : \gamma]` on the appropriate affine plane.

    If the singular point :math:`[\gamma : \alpha : \beta]` is on the
    line at infinity (i.e. :math:`\gamma = 0`) then make the appropriate
    transformation to the curve :math:`f(x,y) = 0` so we can compute
    Puiseux series at the point.

    Parameters
    ----------
    f : sympy.Expr
    x : sympy.Symbol
    y : sympy.Symbol
    singular_pt : list
        A projective singular point of the curve.

    Returns
    -------
    sympy.Expr, sympy.Symbol, sympy.Symbol, complex, complex
        Returns the transformed curve, :math:`g(u,v)` along with the
        variable :math:`u, v` and the centers :math:`u_0, v_0`.

    Examples
    --------
    For example, let :math:`F(x,y,z) = 0` be the homogenized curve.  If
    :math:`beta \neq 0` then the transformation is

    .. math::

        g(u,v) = F(u,beta,v), u0=\alpha, v0=\gamma.

    """
    _z = sympy.Symbol('z')
    alpha, beta, gamma = singular_pt
    F, d = homogenize(f,x,y,_z)

    if gamma == 1:
        return f,x,y,alpha,beta
    else:
        if alpha == 0:
            g = F.subs(y,beta)
            return g,x,_z,alpha,gamma
        else:
            g = F.subs(x,alpha)
            return g,y,_z,beta,gamma



def _multiplicity(P):
    r"""Computes the multiplicity of the singularity at :math:`u_0,v_0`.

    The singularity is given on an affine plane along with the curve
    recentered at the point.

    For each (parametric) Puiseux series

    .. math::

        P_j = \{ x = x(t),  y = y(t) \}

    at :math:`(1 : \alpha : \beta) the contribution from :math:`P_j` to
    the multiplicity is the minimum of the degree of x and the degree of
    y.

    Parameters
    ----------
    P : list
        A list of PuiseuxTSeries of `g = g(u,v)` centered at some
        `(u_0,v_0)` where `g` is a complex affine algebraic curve.

    Returns
    -------
    int
        The multiplicity of the singularity :math:`(u_0, v_0)`.

    """
    m = 0
    for Pi in P:
        n,alpha = zip(*Pi.terms)
        ri = abs(Pi.ramification_index)
        si = abs(min([ni for ni in n if ni != 0]))
        m += min(ri,si)
    return sympy.S(m)


def _branching_number(P):
    r"""Computes the branching number of the singularity at :math:`u_0,v_0`.

    The braching number is simply the number of distinct branches
    (i.e. non-interacting branches) at the place. In parametric form,
    this is simply the number of Puiseux series at the place.

    Parameters
    ----------
    P : list
        A list of PuiseuxTSeries of `g = g(u,v)` centered at some
        `(u_0,v_0)` where `g` is a complex affine algebraic curve.

    Returns
    -------
    int
        The branching number of the singularity :math:`(u_0, v_0)`.

    """
    return sympy.S(len(P))


def _delta_invariant(P):
    r"""Computes the delta invariant of the singularity at :math:`u_0,v_0`.

    Parameters
    ----------
    P : list
        A list of PuiseuxTSeries of `g = g(u,v)` centered at some
        `(u_0,v_0)` where `g` is a complex affine algebraic curve.

    Returns
    -------
    int
        The delta invariant of the singularity :math:`(u_0, v_0)`.

    """
    # compute the puiseux series at (u0,v0). get the parametric forms as
    # well as an x-representative of each parametric form
    # Px = [p.xseries(all_conjugates=True) for p in P]
    # Px_all = [item for sublist in Px for item in sublist]
    Px = [p.xseries(all_conjugates=False)[0] for p in P]
    Px_all = [p.xseries(all_conjugates=True) for p in P]
    Px_all = [item for sublist in Px_all for item in sublist]

    # for each place compute its contribution to the delta invariant
    delta = sympy.Rational(0,1)
    for i in range(len(Px)):
        # compute Int of each x-representation. note that it's
        # sufficient to only look at the Puiseux series with the given
        # v0 since, for all other puiseux series, the contribution to
        # Int() will be zero.
        Pxi = Px[i]
        j = Px_all.index(Pxi)
        IntPxi = Int(j,Px_all)

        # obtain the ramification index by retreiving the corresponding
        # parametric form. By definition, this parametric series
        # satisfies Y(t=0) = v0
        ri = Pxi.ramification_index
        delta += sympy.Rational(ri * IntPxi - ri + 1, 2)
    return sympy.numer(delta)


def genus(f, x, y):
    """Returns the genus of the Riemann surface given by :math:`f=f(x,y)`.

    Uses the singularity structure of the curve to compute the
    genus. This algebraic approach is used as confirmation for the
    geometric approach done in the homology computation in
    `riemann_surface_path_factory`.

    Parameters
    ----------
    f, x, y : sympy.Expr, sympy.Symbol
        An affine algebraic curve.

    Returns
    -------
    int

    """
    _z = sympy.Symbol('z')
    F,d = homogenize(f,x,y,_z)
    S = singularities(f, x, y)
    g = (d-1)*(d-2) / 2
    for pt,(m,delta,r) in S:
        g -= delta
    return g


if __name__ == '__main__':
    print '=== Module Test: singularities.py ==='
    from sympy.abc import x,y

    f1 = (x**2 - x + 1)*y**2 - 2*x**2*y + x**4
    f2 = y**3 + 2*x**3*y - x**7
    f3 = (y**2-x**2)*(x-1)*(2*x-3) - 4*(x**2+y**2-2*x)**2
    f4 = y**2 + x**3 - x**2
    f5 = (x**2 + y**2)**3 + 3*x**2*y - y**3
    f6 = y**4 - y**2*x + x**2
    f7 = y**3 - (x**3 + y)**2 + 1
    f8 = x**6*y**3 + 2*x**3*y - 1
    f9 = 2*x**7*y + 2*x**7 + y**3 + 3*y**2 + 3*y
    f10= (x**3)*y**4 + 4*x**2*y**2 + 2*x**3*y - 1

    f = f10
    import cProfile, pstats
    cProfile.run("s = singularities(f,x,y)",'singularities.profile')
    p = pstats.Stats('singularities.profile')
    p.strip_dirs()
    p.sort_stats('time').print_stats(15)
    p.sort_stats('cumulative').print_stats(15)
    p.sort_stats('calls').print_stats(15)
    for si in s:
        sympy.pprint(si, use_unicode=False)


    # fs = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10]

    # print '\nSingular points of curves:'
    # for i in range(len(fs)):
    #     f = fs[i]
    #     print '\n\tCurve #%d:'%(i+1)
    #     sympy.pprint(f)
    #     print '\nall singular points:'
    #     singular_pts = singularities(f,x,y)
    #     for singular_pt in singular_pts:
    #         print "Point:"
    #         sympy.pprint(singular_pt[0])
    #         sympy.pprint(singular_pt[1])
