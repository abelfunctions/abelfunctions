"""
Differentials
=============

A module for computing the holomorphic differentials defined the on the whole
of the Riemann surface corrsponding to a complex plan algebraic curve.

Authors
-------

- Chris Swierczewski (initial version, October 2012)

"""

import sympy
import pdb

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


def _singularities_finite(f,x,y):
    """
    Returns the finite singularities of f.
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


def _singularities_infinite(f,x,y):
    """
    Returns the singularities of f at infinity.
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


def singularities(f,x,y):
    """
    Compute the singularities of f.

    Returns a list [..., (xi,yi,zi),...] where P is the location of the
    point in the projective plane P^2_C and mP is the multiplicity of
    the singularity.
    """
    S = _singularities_finite(f,x,y)
    S_oo = _singularities_infinite(f,x,y)
    S.extend(S_oo)

    return S

   
if __name__ == '__main__':
    print '=== Module Test: differentials.py ==='
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
    fs = [f5]

    print '\nSingularities of curves:'
    for f in fs:
        print '\n\tCurve:'
        sympy.pprint(f)
        print '\tSingularities:'
        sympy.pprint(singularities(f,x,y))
