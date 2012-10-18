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
import pdb

import sympy

from puiseux import puiseux
from integralbasis import integral_basis

# temporary, hidden symbol to maintain clean Sympy cache
_z = sympy.Symbol('z')

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
    for xk,deg in res.all_roots(multiple=False,radicals=False):
        if deg > 1:
            fxk = sympy.Poly(f.subs({x:xk}), y)
            for ykj,_ in fxk.all_roots(multiple=False,radicals=False):
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
    solsX1 = F0.subs({x:1}).all_roots(multiple=False,radicals=False)
    solsY1 = F0.subs({y:1}).all_roots(multiple=False,radicals=False)

    sols   = [ (1,yi,0) for yi,_ in solsX1 ]
    sols.extend( [ (xi,1,0) for xi,_ in solsY1 ] )

    grad = [F.diff(var) for var in [x,y,_z]]
    for xi,yi,zi in sols:
        if not any( map(lambda e: e.subs({x:xi,y:yi,_z:zi}),grad) ):
            S.append((xi,yi,zi))

    return S


def singular_points(f,x,y):
    """
    Returns the points in P^2_C at which f = f(x,y) is singular.

    Returns a list [..., (xi,yi,zi),...] where P is the location of the
    point in the projective plane P^2_C.
    """
    S = _singular_points_finite(f,x,y)
    S_oo = _singular_points_infinite(f,x,y)
    S.extend(S_oo)

    return S



def branching_number(f,x,y,singular_pt):
    """
    Returns the branching number of a singular point on the 
    complex plane curve f(x,y) = 0.
    """
    pdb.set_trace()

    # compute the Puiseux series at the singular point
    t = sympy.Symbol('t')
    alpha, beta, finite = singular_pt
    if finite:
        P = puiseux(f,x,y,alpha,nterms=1,parametric=t)
    else:
        P = puiseux(f,x,y,sympy.oo,nterms=1,parametric=t)

    # for each of the Pusieux series P = (X(t),Y(t)) pick out the ones
    # with Y(0) == beta and sum the absolute values of the degrees of
    # X(t) for each such series.
    R = 0
    for X,Y in P:
        if Y.subs({t:0}) == beta:
            R += abs(sympy.degree(X,t))
            
    return R


def branching_numbers(f,x,t,singular_pts):
    """
    An efficient way of computing the branching number of multiple
    singular points. Computes only one Puiseux series per x-coordinate
    represented in the list of homogenous singular points.
    """
    return 0
    

   
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
    f8 = x**2*y**6 + 2*x**3*y**5 - 1
    f9 = 2*x**7*y + 2*x**7 + y**3 + 3*y**2 + 3*y
    f10= (x**3)*y**4 + 4*x**2*y**2 + 2*x**3*y - 1
    

    fs = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10]

    print '\nSingular points of curves:'
    for f in fs:
        print '\n\tCurve:'
        sympy.pprint(f)
        print '\n[singular point, branching number]'
        singular_pts = singular_points(f,x,y)
        for singular_pt in singular_pts:
            branching_num = branching_number(f,x,y,singular_pt)
            sympy.pprint([singular_pt, branching_num])
