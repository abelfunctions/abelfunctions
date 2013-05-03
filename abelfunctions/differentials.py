"""
Differentials

This module contains functions for computing a basis of holomorphic
differentials of a Riemann surface given by a complex plan algebraic
curve `f \in C[x,y]`.

A differential `\omega = h(x,y)dx` defined on a Riemann surface `X` is
holomorphic on `X` if it is holomorphic at every point on `X`.
"""

import sympy
import sympy.mpmath as mpmath

from abelfunctions.integralbasis import integral_basis
from abelfunctions.singularities import singularities, _transform, genus
from abelfunctions.utilities import cached_function

import pdb

def mnuk_conditions(f,x,y,b,P):
    """
    Determine the Mnuk conditions on the coefficients, c, of the general
    adjoint polynomial P at the point x=0.

    Note: it is assume t
    """
    if isinstance(b,int): b = sympy.S(b)
    numer, denom = b.ratsimp().as_numer_denom()

    # reduce b*P modulo f
    expr = numer.as_poly(x)*P.as_poly(x)
    Q,R = sympy.polytools.reduced(expr.as_poly(y), [sympy.poly(f,y)])

    # divide by the largest power of x appearing in the denominator.
    # this is sufficient since we've shifted the curve and its
    # singularity to appear at 
    denom = sympy.poly(denom, x)
    try:    mult = sympy.roots(denom)[0]
    except: mult = 0
    rem = expr.as_poly(x).rem(sympy.poly(x**mult,x))
    coeffs = rem.as_poly(x,y).coeffs()

    return coeffs


def differentials(f,x,y):
    """
    Returns a basis of the holomorphic differentials defined on the
    Riemann surface `X: f(x,y) = 0`.

    Input:

    - f: a Sympy object describing a complex plane algebraic curve.
    
    - x,y: the independent and dependent variables, respectively.
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

    # for each singular point [x:y:z] = [alpha:beta:gamma], map f onto
    # the "most convenient and appropriate" affine subspace, (u,v),
    # and center at u=0. determine the conditions on P
    S = singularities(f,x,y)
    conditions = []
    for (alpha, beta, gamma),(m,delta,r) in S:
        g,u,v,u0,v0 = _transform(f,x,y,(alpha,beta,gamma))
        b = integral_basis(g,u,v)
        g = g.subs(u,u+u0)
        for bi in b:
            conditions_bi = mnuk_conditions(g,u,v,bi.subs(u,u+u0),P)
            conditions.extend(conditions_bi)

    # solve the system of equations and retreive monomials contained
    # in the general solution
    c = [item for sublist in c for item in sublist]
    sols = sympy.solve(conditions, c)
    P = P.subs(sols).as_poly(x,y)
    differentials = [x**i * y**j for (i,j) in P.monoms()]

    # sanity check: the number of differentials matches the genus
    g = genus(f,x,y)
    if g != len(differentials):
        raise AssertionError("Number of differentials does not match genus.")


    return [differential/sympy.diff(f,y) for differential in differentials]




if __name__=='__main__':
    print '=== Module Test: differentials.py ==='
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
    f11= y**3 - x**3*y + 2*x**7

    fs = [f1,f2,f3,f4,f5,f6,f9,f10]
    fs = [f11]
    
    for f in fs:
        print "Curve: f(x,y) = %s"%(f)
        print differentials(f,x,y)

