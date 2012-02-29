from __future__ import division

"""
Puiseux Series
==============

"""

###############################################################################
#
# 
#
###############################################################################

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
    testslope = -JJ/II
    
    for (i,j) in hull_with_bdry:
        # when the point is on the j-axis, the onle one we want is the
        # point (0,JJ)
        if i == 0:
            if j == JJ: slope = -sympy.oo
            else:       slope = sympy.oo
        else:           slope = (j-JJ)/i     
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
    q = slope.q; m = -slope.p; l = s*p1[1] - r*p1[0]

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
    T = sympy.symbols('T')
    
    # obtain some coefficient data and compute the newton
    # polygon of f. (The lower convex hull of the support.)
    aij = _coefficient_data(f)
    support = a.keys()
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
        baseslope = (side[1][1]-side[0][1]) / (side[1][0]-side[0][0])
        k = 2

        # check against all following points for colinearity by comparing
        # slopes. append all colinear points to side
        for kk in xrange(2,N-n-1):
            pt = newton[n+k]
            slope = (pt[1]-size[0][1]) / (pt[0]-size[0][0])
            if abs(slope - base_slope) < eps:
                side.append(pt)
            else:
                k = kk
                break
        n += k-1

        # now that we have the side, compute the q,m,l,u,v data
        # and the characteristic polynomial
        q,m,l,u,v = _qmluv(side)
        i0 = min(side, key=itemgetter(0))[0]
        phi = sum(aij[(i,j)]*T**Rational(i-i0,q) for (i,j) in side)
        

            


    

"""
TESTS
"""
if __name__ == "__main__":
    print "==== Test Suite: puiseux.py ==="

    # example algebraic curve
    x,y = sympy.symbols('x,y')
    f   = sympy.poly(x**4 + x**2*y**2 - 2*x**2*y - y**2*x + y**2)
    #f = sympy.poly(-y**5+(2*x-1)*y**4+(3*x-x**2)*y**3+(x-3*x**2)*y**2+(x**3-2*x**2)*y+x**6)
    
    # display coefficient data, support, and newton polygon
    a = _coefficient_data(f)
    support = a.keys()
    newton = newton_polygon(support)
    print "Newton Polygon:"
    print "  f =", f
    print "  a =", a
    print
    print "  support =", support
    print "  newton  =", newton
    print "  Polygon(newton) =", sympy.Polygon(*newton)
    

