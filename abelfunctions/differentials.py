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

def mnuk_conditions(f,u,v,b,P):
    """
    Determine the Mnuk conditions on the coefficients, c, of the general
    adjoint polynomial P at the point x=0.

    Note: it is assume t
    """
    numer, denom = b.ratsimp().as_numer_denom()

    # reduce b*P modulo f
    expr = (numer*P).as_poly(v,u)
    Q,R = sympy.polytools.reduced(expr, [sympy.poly(f,v,u)])

    # divide by the largest power of x appearing in the denominator.
    # this is sufficient since we've shifted the curve and its
    # singularity to appear at
    try:
        mult = sympy.roots(denom.as_poly(u))[sympy.S(0)]
    except KeyError:
        mult = 0

    R = R.as_poly(u,v)
    coeffs = R.coeffs()
    monoms = R.monoms()
    conditions = [coeff for coeff,monom in zip(coeffs,monoms)
                  if monom[0] < mult]
    return conditions


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
    for singular_pt,(m,delta,r) in S:
        g,u,v,u0,v0      = _transform(f,x,y,singular_pt)
        Ptilde,u,v,u0,v0 = _transform(P,x,y,singular_pt)
        g = g.subs(u,u+u0)
        
        b = integral_basis(g,u,v)
        for bi in b:
            conditions_bi = mnuk_conditions(g,u,v,bi,Ptilde)
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

    f = f11
    
    import cProfile, pstats
    cProfile.run(
        'D = differentials(f,x,y)',
        'differentials.profile',
        )
    p = pstats.Stats('differentials.profile')
    p.strip_dirs()
    p.sort_stats('time').print_stats(15)
    p.sort_stats('cumulative').print_stats(15)
    p.sort_stats('calls').print_stats(15)

    print "\nDifferentials:"
    for omega in D:
        sympy.pretty_print(omega)
        print
