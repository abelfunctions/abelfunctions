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

from integralbasis import integral_basis
from singularities import singularities, _transform
from utilities import cached_function



def _eliminate_higher_orders(h,x,y,deg,expr):
    """
    Given a rational function `h = h(x,y) \in C(x)[y]`, replace all powers
    of `y**deg` with `expr`.
    """    
    m = sympy.degree(h,y)
    h = h.expand()
    hred = sympy.S(0)

    for term in h.as_ordered_terms():
        term_deg = sympy.degree(term,y)
        term = term.replace(sympy.Pow, 
                            lambda arg, pow: arg**(pow-deg)*expr
                            if arg == y and pow >= deg 
                            else arg**pow)
        hred += term

    return hred.expand()



def _solve_for_leading_y(f,x,y):
    """
    Given `f(x,y) = a_n(x) * y**n + ... + a_0(x)`, return
    
        (-a_{n-1}(x) - ... - a_o(x)) / a_n(x)

    """
    n = sympy.degree(f,y)
    sol = sympy.S(0)
    a_n = sympy.S(1)

    for term in f.expand().as_ordered_terms():
        if sympy.degree(term,y) == n:
            a_n = term.coeff(y**n)
        else:
            sol -= term

    return sol/a_n
            

def _adjoint_monomials(f,x,y,bi,P,c):
    """
    Obtain the adjoint monomials corresponding to the integral basis
    element bi. This is done by demanding that

        bi * P

    be a polynomial where

        P = sum( c_{ij} x**i y**j )
    """
    f = f.expand()
    n = sympy.degree(f,y)
    d = f.as_poly().total_degree()

    rhs = _solve_for_leading_y(f,x,y)
    Pi = bi * P
    Pi = _eliminate_higher_orders(Pi,x,y,n,rhs)

    p = sympy.Wild('p')
    q = sympy.Wild('q')
    coeff = sympy.Wild('coeff')

    monoms = set([])
    for term in Pi.as_ordered_terms(): 
        if sympy.degree(x) < 0:
            # strips out the cij from the monomial
            cij = term.as_independent(x,y)[0].free_symbols.pop()
            i,j = c[cij]
            monoms.add(x**i*y**j)


def differentials(f,x,y):
    """
    Returns a basis of the holomorphic differentials defined on the
    Riemann surface `X: f(x,y) = 0`.

    Input:

    - f: a Sympy object describing a complex plane algebraic curve.
    
    - x,y: the independent and dependent variables, respectively.
    """
    d = f.as_poly().total_degree()
    n = sympy.degree(f,y)

    # coeffiecients and general adjoint polynomial
    c_arr = sympy.symarray('c',(d-3,d-3)).tolist()
    c = dict( (cij,(c_arr.index(ci),ci.index(cij))) 
              for ci in c_arr for cij in ci )
    P = sum( c_arr[i][j]*x**i*y**j 
             for i in range(d-3) for j in range(d-3) if i+j <= d-3)

    S = singularities(f,x,y)
    differentials = set([])
    for (alpha, beta, gamma), (m,delta,r) in S:
        g,u,v,u0,v0 = _transform(f,x,y,(alpha,beta,gamma))

#        if delta > m*(m-1)/2:
        if True:
            # Use integral basis method.
            b = integral_basis(g,u,v)
            for bi in b:
                monoms = _adjoint_monomials(f,x,y,bi,P,c)
                differentials.add(monom for monom in monoms) 
        else:
            # Use Puiseux series method
            b = integral_basis(g,u,v)


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

    f = f7

#     for f in fs:
#         print '\nCurve:'
#         sympy.pprint(fs)

#         print '\nSingularities:'
#         S = singularities(f,x,y)
#         for (alpha,beta,gamma),(m,delta,r) in S:
#             print '\nSingular Point:'
#             sympy.pprint((alpha,beta,gamma))
#             sympy.pprint((m,delta,r))

#             print '\nTransformed Curve:'
#             g,u,v,u0,v0 = _transform(f,x,y,(alpha,beta,gamma)) 
#             sympy.pprint(g)
            
        
