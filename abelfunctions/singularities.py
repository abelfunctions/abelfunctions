r"""Singularities :mod:`abelfunctions.singularities`
================================================

A module for computing the singular points of a complex plane algebraic curve
including their multiplicities, branching numbers, multiplicities, and delta
invariants.

Each singularity :math:`P = (x,y,z)` on the projectivization of the curve has
associated with it a three-tuple :math:`(m, \delta, r)` representing the
multiplicity, delta invariant, and branching number of the multiplicity.

Functions
---------

singularities

Examples
--------

Contents
--------
"""
import sympy
from sympy import RootOf, Rational, Poly

from .puiseux import puiseux
from .integralbasis import Int
from .utilities import rootofsimp, cached_function


def singular_points_finite(f,x,y):
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

    # the discriminant points (roots of resultant) contain the x-points where
    # singularities may occur. todo: reuse RiemannSurface.discriminant_points()
    p = f.as_poly(x,y)
    n  = p.degree(y)
    res = sympy.resultant(p,p.diff(y),y).as_poly(x)
    xroots = res.all_roots(multiple=False, radicals=False)
    for xk,deg in xroots:
        if deg > 1:
            # evaluate p at xk. we use xreplace in case xk is a RootOf. (Sympy
            # does not always preserve radical=True)
            pxk = p.as_expr().xreplace({x:xk})
            pxk = rootofsimp(pxk).as_poly(y)

            # compute y-roots at x=xk. an error will be thrown if RootOfs still
            # appear in p(x=xk). in this case, use the algebraic expressions of
            # the roots
            try:
                yroots = pxk.all_roots(multiple=False,radicals=False)
            except NotImplementedError:
                yroots = sympy.roots(pxk,y).items()

            # for each y-root ykj above xk, record a singular point if the
            # gradient vanishes at (xk,ykj)
            for ykj,_ in yroots:
                subs = {x:xk,y:ykj}
                dfdx = rootofsimp(f.diff(x).subs(subs))
                dfdy = rootofsimp(f.diff(y).subs(subs))

                # if there are any leftover RootOfs we should approximate the
                # vanishing numerically. this is done to handle situations such
                # as I + (-I) = 0
                def is_zero(expr):
                    if expr.has(RootOf):
                        return abs(dfdx.n(n=18)) < 1e-16
                    else:
                        return rootofsimp(expr).simplify() == 0

                if is_zero(dfdx) and is_zero(dfdy):
                    S.append((xk,ykj,sympy.S(1)))
    return S


def singular_points_infinite(f,x,y,z):
    r"""Returns the singular points of `f` at infinity.

    In particualr, returns all of the projective singular points
    :math:`(x_k,y_k,z_k)` such that :math:`z_k = 0`.

    Parameters
    ----------
    f, x, y : sympy.Expr, sympy.Symbol
        An affine algebraic curve.
    z : sympy.Symbol

    Returns
    -------
    list
        A list of three-tuples :math:`(x_k,y_k,0)` representing the infinite
        singular points of the curve :math:`f = f(x,y)`

    """
    # compute the homogenization of degree d
    F = f.as_poly(x,y).homogenize(z)
    d = F.total_degree()

    # find the possible singular points at infinity. these consist of the roots
    # of F(1,y,0) = 0 and F(x,1,0) = 0
    F0 = F.subs(z,0)
    F0x1 = F0.subs({x:1,y:z}).as_poly(z)
    F0y1 = F0.subs({x:z,y:1}).as_poly(z)
    solsx1 = F0x1.all_roots(multiple=False,radicals=False)
    solsy1 = F0y1.all_roots(multiple=False,radicals=False)
    all_points = [(1,yi,0) for yi,_ in solsx1]
    all_points.extend([(xi,1,0) for xi,_ in solsy1])

    # these possible singularities are in projective space, so filter out equal
    # points such as (0,1,I) == (0,-I,1). normalize these projective points
    # such that 1 appears in either the x- or y- coordinate, where appropriate
    normalized_points = []
    zero = sympy.S(0)
    for xi,yi,zi in all_points:
        P = (1,yi/xi,0) if xi != zero else (xi/yi,1,0)
        if not P in normalized_points:
            normalized_points.append(P)

    # finally, check the gradient condition to get the actual singular points
    S = []
    grad = [F.diff(var) for var in (x,y,z)]
    for xi,yi,zi in normalized_points:
        grad_vals = [dFi.subs({x:xi,y:yi,z:zi}) for dFi in grad]
        grad_vals = map(rootofsimp, grad_vals)

        # if there are any leftover RootOfs we should approximate the vanishing
        # numerically. this is done to handle situations such as I + (-I) = 0
        #
        # [XXX] the following is messy but works.
        is_zero = True
        for val in grad_vals:
            if val.has(RootOf):
                val_is_zero = abs(val.n(n=18)) < 1e-16
            else:
                val_is_zero = val == 0
            if not val_is_zero:
                is_zero = False
                break

        if is_zero:
            S.append((xi,yi,zi))
    return S


@cached_function
def singularities(f,x,y):
    r"""Returns the singularities of the curve `f` in projective space.

    Returns all of the projective singular points :math:`(x_k,y_k,z_k)` on the
    curve :math:`f(x,y) = 0`. For each singular point a three-tuple :math:`(m,
    \delta, r)` is given representing the multiplicity, delta invariant, and
    branching number of the singularity.

    This information is used to resolve singularities for the purposes of
    computing a Riemann surface. The singularities are resolved using Puiseux
    series.

    Parameters
    ----------
    f : sympy.Expr
    x : sympy.Symbol
    y : sympy.Symbol

    Returns
    -------
    list
        A list of the singularities, both finite and infinite, along with their
        multiplicity, delta invariant, and branching number information.

    """
    z = sympy.Dummy('z')
    S = singular_points_finite(f,x,y)
    S_oo = singular_points_infinite(f,x,y,z)
    S.extend(S_oo)

    info = []
    for singular_pt in S:
        # Perform a projective transformation of the curve so it's almost
        # centered at the singular point.
        g,u,v,u0,v0 = _transform(f,x,y,z,singular_pt)
        P = puiseux(g,u,v,u0,v0)

        # filter out any places with infinite v-part: they are being handled by
        # other centerings / transformations
        def has_finite_v(Pi):
            # make sure the order of the y-part is positive
            n,alpha = zip(*Pi.terms)
            while Pi.order <= 0:
                Pi.add_term()

            # if there are still no terms then they are positive exponent
            if n == []:
                return True
            elif min(n) >= 0:
                return True
            return False

        P = filter(has_finite_v,P)
        m = _multiplicity(P)
        delta = _delta_invariant(P)
        r = _branching_number(P)

        info.append((m,delta,r))

    return zip(S,info)


def _transform(f,x,y,z,singular_pt):
    r"""Recenters the affine curve `f` at a singular point.

    Returns :math:`(g,u,v,u0,v0)` where :math:`g = g(u,v)` is the transformed
    polynomial and :math:`u0,v0` is the projection of :math:`[\alpha : \beta :
    \gamma]` on the appropriate affine plane.

    If the singular point :math:`[\gamma : \alpha : \beta]` is on the line at
    infinity (i.e. :math:`\gamma = 0`) then make the appropriate transformation
    to the curve :math:`f(x,y) = 0` so we can compute Puiseux series at the
    point.

    Parameters
    ----------
    f : sympy.Expr
    x : sympy.Symbol
    y : sympy.Symbol
    z : sympy.Symbol
    singular_pt : list
        A projective singular point of the curve.

    Returns
    -------
    sympy.Expr, sympy.Symbol, sympy.Symbol, complex, complex
        Returns the transformed curve, :math:`g(u,v)` along with the variable
        :math:`u, v` and the centers :math:`u_0, v_0`.

    Examples
    --------
    For example, let :math:`F(x,y,z) = 0` be the homogenized curve.  If
    :math:`beta \neq 0` then the transformation is

    .. math::

        g(u,v) = F(u,beta,v), u0=\alpha, v0=\gamma.

    """
    alpha, beta, gamma = singular_pt
    F = f.as_poly(x,y).homogenize(z)
    d = F.total_degree()

    if gamma == 1:
        return f,x,y,alpha,beta
    else:
        if alpha == 0:
            g = F.subs(y,beta)
            return g,x,z,alpha,gamma
        else:
            g = F.subs(x,alpha)
            return g,y,z,beta,gamma



def _multiplicity(P):
    r"""Computes the multiplicity of the singularity at :math:`u_0,v_0`.

    The singularity is given on an affine plane along with the curve recentered
    at the point.

    For each (parametric) Puiseux series

    .. math::

        P_j = \{ x = x(t),  y = y(t) \}

    at :math:`(1 : \alpha : \beta) the contribution from :math:`P_j` to the
    multiplicity is the minimum of the degree of x and the degree of y.

    Parameters
    ----------
    P : list
        A list of PuiseuxTSeries of `g = g(u,v)` centered at some `(u_0,v_0)`
        where `g` is a complex affine algebraic curve.

    Returns
    -------
    int
        The multiplicity of the singularity :math:`(u_0, v_0)`.

    """
    m = 0
    for Pi in P:
        n,alpha = zip(*Pi.terms)
        ri = abs(Pi.ramification_index)

        # get the leading order behavior of the y-part. we can save time if the
        # order of the y-part exceeds the ramification index, ri
        try:
            si = min([ni for ni in n if ni != 0])
        except ValueError:
            if Pi.order >= ri:
                si = ri
            else:
                # extend to order ri. if there are still no non-zero terms then
                # used the shortcut above
                Pi.extend(ri+1)
                try:
                    si = min([ni for ni in n if ni != 0])
                except ValueError:
                    si = ri

        m += min(ri,si)
    return sympy.S(m)


def _branching_number(P):
    r"""Computes the branching number of the singularity at :math:`u_0,v_0`.

    The braching number is simply the number of distinct branches
    (i.e. non-interacting branches) at the place. In parametric form, this is
    simply the number of Puiseux series at the place.

    Parameters
    ----------
    P : list
        A list of PuiseuxTSeries of `g = g(u,v)` centered at some `(u_0,v_0)`
        where `g` is a complex affine algebraic curve.

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
        A list of PuiseuxTSeries of `g = g(u,v)` centered at some `(u_0,v_0)`
        where `g` is a complex affine algebraic curve.

    Returns
    -------
    int
        The delta invariant of the singularity :math:`(u_0, v_0)`.

    """
    # compute the puiseux series at (u0,v0). get the parametric forms as well
    # as an x-representative of each parametric form
    Px = [p.xseries(all_conjugates=False)[0] for p in P]
    Px_all = [p.xseries(all_conjugates=True) for p in P]
    Px_all = [item for sublist in Px_all for item in sublist]

    # for each place compute its contribution to the delta invariant
    delta = sympy.Rational(0,1)
    for i in range(len(Px)):
        # compute Int of each x-representation. note that it's sufficient to
        # only look at the Puiseux series with the given v0 since, for all
        # other puiseux series, the contribution to Int() will be zero.
        Pxi = Px[i]
        j = Px_all.index(Pxi)
        IntPxi = Int(j,Px_all)

        # obtain the ramification index by retreiving the corresponding
        # parametric form. By definition, this parametric series satisfies
        # Y(t=0) = v0
        ri = Pxi.ramification_index
        delta += sympy.Rational(ri * IntPxi - ri + 1, 2)
    return sympy.numer(delta)


@cached_function
def genus(f,x,y):
    """Returns the genus of the Riemann surface given by :math:`f=f(x,y)`.

    Uses the singularity structure of the curve to compute the genus. This
    algebraic approach is used as confirmation for the geometric approach done
    in the homology computation in `riemann_surface_path_factory`.

    Parameters
    ----------
    f, x, y : sympy.Expr, sympy.Symbol
        An affine algebraic curve.

    Returns
    -------
    int

    """
    z = sympy.Dummy('z')
    F = f.as_poly(x,y).homogenize(z)
    d = F.total_degree()
    S = singularities(f,x,y)
    g = (d-1)*(d-2) / 2
    for pt,(m,delta,r) in S:
        g -= delta
    return g
