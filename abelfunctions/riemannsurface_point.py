"""
Riemann Surface Points

A module for defining points on a Riemann surface.
"""

import numpy
import scipy
import sympy


class RiemannSurfacePoint():
    """
    A lightweight class defining a point on a Riemann surface.
    """
    def __init__(self, RS, x, y, exact=False):
        """
        Input:
        
        - `RS`: a RiemannSurface object

        - `x,y`: the x- and approximate y- coordinates of the point.
        A multiprecise root-finder is used to narrow in on a y.

        - `exact` (default: False): If false, use a multiprecise root
        finder to get a closer y-value for the point. If True, assumes
        that the input y-value is within mpmath.eps of the true value.

        TODO: Add infrastructure for points at x=infinity.
        """
        self.RS = RS
        self.x = x
        
        # use root finder to obtain closer y-value unless user
        # specifies that they obtained their result using a
        # multiprecise root-finder.
        if exact:
            self.y = y
        else:
            fx     = lambda z: RS.f(x,z)
            yp     = - RS.dfdx(x,y) / dfdy(x,y)
            guess  = (y-yp, y, y+yp)
            self.y = sympy.mpmath.findroot(fx, guess, solver='muller')


    def __repr__(self):
        return "(%s, %s)" %(self.x, self.y)


    def __eq__(self,other):
        """
        TODO: add way of determining equality of Riemann surfaces, if
        possible.
        """
        almosteq = sympy.mpmath.almosteq

        if isinstance(other,RiemannSurfacePoint):
            if almosteq(self.x,other.x) and almosteq(self.y,other.y):
                return True
        return False

            
    def __hash__(self):
        return hash((self.x, self.y))
    
