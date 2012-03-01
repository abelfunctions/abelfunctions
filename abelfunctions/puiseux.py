#from __future__ import division

"""
Puiseux Series
==============

Tools for computing Puiseux series. A necessary component to computing 
integral bases and with Riemann surfaces.
"""

import numpy
import scipy
import sympy
from operator import itemgetter


def _coefficient_data(f):
    """
    Helper function. Returns a dictionary of coefficients of the polynomial
    indexed by monomial powers ..math :

        `a_{ij} x^j y^i`

    INPUTS:
    
        -- ``f``: a sympy polynomial

    OUTPUTS:

        -- ``dict``: a dictionary such that ``d[(i,j)] = a_ij``.
    """
    # compute useful dictionary of coefficients indexed
    # by the support of the polynomial
    d = {}
    monoms = f.monoms()
    coeffs = f.coeffs()
    for a,(j,i) in zip(coeffs,monoms):
        d[(i,j)] = a
    return d


def newton_polygon(support):
    """
    Computes the Newton polygon of a convex polygon. In this case, the Newton
    polygon is defined as the "lower convex hull" of the support of a 
    polynomial, that is, the collection of all edges "facing" the origin.
    """
    # compute the lower convex hull of f.
    #
    # since sympy.convex_hull doesn't include points on edges, we need
    # to compare back to the support. the convex hull will contain all
    # points. Those on the boundary are considered "outside" by sympy.
    hull = sympy.convex_hull(*support)
    hull_with_bdry = [p for p in support if not hull.encloses(sympy.Point(p))]
    newton = []
    
    # find the start and end points (0,J) and (I,0)
    JJ = min([j for (i,j) in hull if i == 0])
    II = min([i for (i,j) in hull if j == 0])
    testslope = -float(JJ)/II
    
    for (i,j) in hull_with_bdry:
        # when the point is on the j-axis, the onle one we want is the
        # point (0,JJ)
        if i == 0:
            if j == JJ: slope = -sympy.oo
            else:       slope = sympy.oo
        else:           slope = float(j-JJ)/i     
        # equality is for the case when the point is (II,0)
        if slope <= testslope: newton.append((i,j))
            
    newton.sort(key=itemgetter(1),reverse=True) # sort second in j-th coord
    newton.sort(key=itemgetter(0))              # sort first in i-th coord
    return newton


def _qmluv(side):
    # get the endpoints of the side
    p1,p2 = side[0], side[-1]

    # determine q,m,l from slope
    slope = sympy.Rational(p2[1]-p1[1], p2[0]-p1[0])
    q = slope.q; m = -slope.p
    l = q*p1[1] + m*p1[0]

    # use xgcd to compute u,v such that qu-mv=1
    u,v,g = sympy.gcdex(q,-m)
    return q,m,l,u,v


def newton_puiseux(f,x,y):
    """
    Returns a set of triplets `\{[G_i,P_i,Q_i]\}` such that .. math:
        
        F(P_i(X),Q_i(X,Y)) = G_i(X,Y)
        
    and the polynomials `G_i(X,Y)` are regular at `X=0`.
    """
    # create a container for the triplets
    L = []
    eps = 1e-14
    T = sympy.Symbol('T')
    
    # obtain some coefficient data and compute the newton
    # polygon of f. (The lower convex hull of the support.)
    p = sympy.poly(f,x,y)
    aij = _coefficient_data(p)
    support = aij.keys()
    newton = newton_polygon(support)
    
    # loop over each side. compute:
    #   1) q,m,l such that qj+mi=1
    #   2) u,v such that uq-mv=1
    #   3) phi_Delta = characteristic polynomial for the side Delta
    N = len(newton)
    n = 0
    while n < N-1:
        # determine slope of current line
        side = [newton[n],newton[n+1]]
        baseslope = float(side[1][1]-side[0][1]) / (side[1][0]-side[0][0])
        k = 2

        # check against all following points for colinearity by comparing
        # slopes. append all colinear points to side
        for kk in xrange(2,N-n-1):
            pt = newton[n+k]
            slope = float(pt[1]-side[0][1]) / (pt[0]-side[0][0])
            if abs(slope - baseslope) < eps:
                side.append(pt)
            else:
                k = kk
                break
        n += k-1

        # now that we have the side, compute the q,m,l,u,v data
        # and the characteristic polynomial
        q,m,l,u,v = _qmluv(side)
        i0 = min(side, key=itemgetter(0))[0]
        phi = sum(aij[(i,j)]*T**sympy.Rational(i-i0,q) for (i,j) in side)
        rts = sympy.roots(phi)

        for xi,M in rts.iteritems():
            # compute the polynomial H. 
            H = f.subs({x:(xi**v*x**q), y:(xi**u*x**m*(1+y))})
            H = (H/(x**l)).simplify()

            # recurse when we reach a multiple root
            if M == 1: L1 = [[H,x,y]]
            else:      L1 = newton_puiseux(H,x,y)

            # append Puiseux data
            for G,P,Q in L1:
                L.append([G,
                          (xi**v*P**q).simplify(),
                          (xi**u*P**m*(1+Q)).simplify()])

    return L


def puiseux(f,x,y,a,n,T=True):
    """
    Compute the degree `n` Puiseux series of `f`.
    """
    from sympy.abc import T

    # perform shift
    if a == sympy.oo: f = f.subs(x,1/x)
    else:             f = f.subs(x,x+a)

    # stack for keeping track of Puiseux terms
    stack = newton_puiseux(f,x,y)
    series = []

    while len(stack) > 0:
        # 1) pop last element of stack and check degree of series
        G,P,Q = stack.pop()
        p = sympy.poly(P,x,y)
        q = sympy.poly(Q,x,y)
        deg = float(q.degree(x)) / p.degree(x)

        if deg >= n:
            # 2a) if degree exceeds target degree, n, then add to storage
            #     list "series" and pop from stack
            P = P.subs(x,T)
            Q = Q.subs({x:T,y:0})
            series.append((P,Q))
        else:
            # 2b) otherwise, compute terms following these and extend the stack
            next_terms = newton_puiseux(G,x,y)
            extended_series = []
            for G1,P1,Q1 in next_terms:
                ext = [G1,P.subs(x,P1),Q.subs({y:Q1})]
                extended_series.append(ext)
            stack.extend(extended_series)

    return series

        
    

"""
TESTS
"""
if __name__ == "__main__":
    print "==== Test Suite: puiseux.py ==="
    # example algebraic curve
    x,y,T = sympy.symbols('x,y,T')
    f1 = -y**5 + (2*x-1)*y**4 - (3*x-x**2)*y**3 - (x-3*x**2)*y**2 + \
         (x**3-2*x**2)*y + x**6
    f2 = y**2 - x**2*(x+1)
    f3 = 0
    f  = f1

    print "Curve:"
    print 
    print "\t", f
    print
    
    # display coefficient data, support, and newton polygon
    a = _coefficient_data(sympy.Poly(f))
    support = a.keys()
    newton = newton_polygon(support)
    print "Newton Polygon:"
    print "  support =", support
    print "  newton  =", newton
    print
    

    # compute leading terms
    print "Leading terms in Puiseux series:"
    L = newton_puiseux(f,x,y)
    for G,P,Q in L:
        print "  P =", P.subs(x,T)
        print "  Q =", Q.subs(x,T)
        print "     \\"
        L1 = newton_puiseux(G,x,y)
        for G1,P1,Q1 in L1:            
            print "    P1 =", P1.subs(x,T)
            print "    Q1 =", Q1.subs(x,T)
            print "        \\"
#             L2 = newton_puiseux(G1,x,y)
#             for G2,P2,Q2 in L2:
#                 print "      P2 =", P2.subs(x,T)
#                 print "      Q2 =", Q2.subs(x,T)
#                print

"""
    # compute Puiseux series
    print "Puiseux series:"
    p = puiseux(f,x,y,0,2)
    print
    for p1 in p:
        print "\tP =", p1[0].expand()
        print "\tQ =", p1[1].expand()
        print
"""
