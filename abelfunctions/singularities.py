"""
Singularities
=============

A module for computing the singular points of a complex plane algebraic curve
including their multiplicities, branching numbers, multiplicities, and delta
invariants.

Authors
-------

- Chris Swierczewski (initial version, October 2012)

"""
import sympy

import pdb

from puiseux import puiseux
from integralbasis import Int, valuation
from utilities import cached_function

# temporary, hidden symbol to maintain clean Sympy cache
_t = sympy.Symbol('t')
_z = sympy.Symbol('z')
_Z = sympy.Dummy('Z')


@cached_function
def homogenize(f,x,y,z):
    """
    Returns the polynomial in homogenous coordinates.
    """
    p = sympy.poly(f,[x,y])
    d = max(map(sum,p.monoms()))
    F = sympy.expand( z**d * f.subs([(x,x/z),(y,y/z)]) )
    return F, d


def _singular_points_finite(f,x,y):
    """
    Returns the finite singular points of f.
    """
    S = []

    # compute the finite singularities: use the resultant
    p  = sympy.Poly(f,[x,y])
    n  = p.degree(y)

    res = sympy.Poly(sympy.resultant(p,p.diff(y),y),x)
    for xk,deg in sympy.roots(res,x).iteritems():
        if deg > 1:
            fxk = sympy.Poly(f.subs({x:xk,y:_Z}), _Z)
            for ykj,_ in sympy.roots(fxk, _Z).iteritems():
                fx = f.diff(x)
                fy = f.diff(y)
                subs = {x:xk,y:ykj}
                if (fx.subs(subs) == 0) and (fy.subs(subs) == 0):
                    S.append((xk,ykj,1))

    return S


def _singular_points_infinite(f,x,y):
    """
    Returns the singular points of f at infinity.
    """
    S = []

    # compute homogenous polynomial
    F, d = homogenize(f,x,y,_z)

    # find singular points at infinity
    F0 = sympy.Poly(F.subs([(_z,0)]),[x,y])
    domain = sympy.QQ[sympy.I]
    solsX1 = sympy.roots(F0.subs({x:1,y:_Z}),_Z).keys()
    solsY1 = sympy.roots(F0.subs({y:1,x:_Z}),_Z).keys()

    all_sols = [ (1,yi,0) for yi in solsX1 ]
    all_sols.extend( [ (xi,1,0) for xi in solsY1 ] )

    # these points are in projective space, so filter out equal points
    # such as (1,I,0) == (-I,1,0). We normalize these projective
    # points such that 1 appears in either the x- or y- coordinate
    sols = []
    for xi,yi,zi in all_sols:
        normalized = (1,yi/xi,0) if xi != sympy.S(0) else (xi/yi,1,0)
        if not normalized in sols:
            sols.append(normalized)

    # Filter out any points that are not singular by checking if the
    # gradient vanishes.
    grad = [F.diff(var) for var in [x,y,_z]]
    for xi,yi,zi in sols:
        fsub = lambda e,x=x,y=y,_z=_z,xi=xi,yi=yi,zi=zi:  \
               e.subs({x:xi,y:yi,_z:zi}).simplify() != sympy.S(0)
        if not any(map(fsub,grad)):
            S.append((xi,yi,zi))

    return S


@cached_function
def singularities(f,x,y):
    """
    Returns the points in P^2_C at which f = f(x,y) is singular.

    Returns a list [..., (xi,yi,zi),...] where P is the location of the
    point in the projective plane P^2_C.
    """
    S = _singular_points_finite(f,x,y)
    S_oo = _singular_points_infinite(f,x,y)
    S.extend(S_oo)

    info = []
    for singular_pt in S:
        # Perform a projective transformation of the curve so it's
        # almost centered at the singular point.
        g,u,v,u0,v0 = _transform(f,x,y,singular_pt)

        m     = _multiplicity(g,u,v,u0,v0)
        delta = _delta_invariant(g,u,v,u0,v0)
        r     = _branching_number(g,u,v,u0,v0)

        info.append((m,delta,r))

    return zip(S,info)


def _transform(f,x,y,singular_pt):
    """
    If the singular point [alpha : beta : gamma] is on the 
    line at infinity (i.e. gamma = 0) then make the appropriate
    transformation to the curve f(x,y) = 0 so we can compute Puiseux
    series at the point.

    Returns (g,u,v,u0,v0) where g = g(u,v) is the transformed
    polynomial and u0,v0 is the projection of [alpha : beta : gamma]
    on the appropriate affine plane.

    For example, let F(x,y,z) = 0 be the homogenized polynomial.
    If beta != 0 then the transformation is
    
        g(u,v) = F(u,beta,v), u0=alpha, v0=gamma.

    """
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



def _multiplicity(g,u,v,u0,v0):
    """
    Returns the multiplicity of the place (alpha : beta : 1) from the
    Puiseux series P at the place.

    For each (parametric) Puiseux series
        
        Pj = { x = x(t)
             { y = y(t) 
    
    at (alpha : beta : 1) the contribution from Pj to the multiplicity
    is min( deg x(t), deg y(t) ).
    """
    # compute Puiseux expansions at u=u0 and filter out
    # only those with v(t=0) == v0
    P = puiseux(g,u,v,u0,nterms=0,parametric=_t)

    m = 0
    for X,Y in P:
        X = X - u0                      # Shift so no constant
        Y = Y - v0                      # term remains.
        ri = abs( X.leadterm(_t)[1] )   # Get order of lead term
        si = abs( Y.leadterm(_t)[1] )
        m += min(ri,si)

    return m



def _branching_number(g,u,v,u0,v0):
    """
    Returns the branching number of the place [alpha : beta : 1]
    from the Puiseux series P at the place.
        
    The braching number is simply the number of distinct branches
    (i.e. non-interacting branches) at the place. In parametric form,
    this is simply the number of Puiseux series at the place.
    """
    # compute Puiseux expansions at u=u0 and filter out
    # only those with v(t=0) == v0
    P = puiseux(g,u,v,u0,nterms=1,parametric=_t)
    P_v0 = [(X,Y) for X,Y in P if Y.subs(_t,0) == v0]

    return sympy.S(len(P_v0))



def _delta_invariant(g,u,v,u0,v0):
    """
    Returns the delta invariant corresponding to the singular point
    `singular_pt` = [alpha, beta, gamma] on the plane algebraic curve
    f(x,y) = 0.
    """
    # compute Puiseux expansions at u=u0 and filter out only those
    # with v(t=0) == v0. We only chose one y=y(x) Puiseux series for
    # each place as a representative to prevent over-counting by using
    # the "grouped=True" flag in Puiseux
    P = puiseux(g,u,v,u0,nterms=0,parametric=_t,grouped=True)
    P_x = puiseux(g,u,v,u0,nterms=0,parametric=False,grouped=True)
    P_x_v0 = []
    for i in range(len(P)):
        X,Y = P[i]
        p = P_x[i][0]
        if Y.subs(_t,0).simplify() == v0:
            # store the index as well so we know which parametric form
            # corresponds to this puiseux series.
            P_x_v0.append((p,i))

    # now obtain ungrouped series
    P_x = puiseux(g,u,v,u0,nterms=0,parametric=False)
    
    # for each place compute its contribution to the delta invariant
    delta = sympy.Rational(0,1)
    for i in range(len(P_x_v0)):
        yhat, place_index = P_x_v0[i]
        j = P_x.index(yhat)
        IntPj = Int(j,P_x,u,u0)

        # obtain the ramification index by retreiving the
        # corresponding parametric form. By definition, this
        # parametric series satisfies Y(t=0) = v0
        X,Y = P[place_index]
        rj = (X-u0).as_coeff_exponent(_t)[1]
        delta += sympy.Rational(rj * IntPj - rj + 1, 2)

    return sympy.numer(delta)


@cached_function
def genus(f,x,y):
    """
    Returns the genus of the Riemann surface corresponding to the complex plane
    algebraic curve f = f(x,y).
    """
    F,d = homogenize(f,x,y,_z)
    S = singularities(f,x,y)

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
#    f7 = y**3 - (x**3 + y)**2 + 1
    f8 = x**6*y**3 + 2*x**3*y - 1
    f9 = 2*x**7*y + 2*x**7 + y**3 + 3*y**2 + 3*y
    f10= (x**3)*y**4 + 4*x**2*y**2 + 2*x**3*y - 1
    

#    fs = [f1,f2,f3,f4,f5,f6,f8,f9,f10]
    fs = [f10]

    print '\nSingular points of curves:'
    for f in fs:
        print '\n\tCurve:'
        sympy.pprint(f)
        print '\nall singular points:'
        singular_pts = singularities(f,x,y)
        for singular_pt in singular_pts:
            print "Point:"
            sympy.pprint(singular_pt[0])
            sympy.pprint(singular_pt[1])
