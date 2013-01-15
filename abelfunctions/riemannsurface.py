"""
Riemann Surfaces
"""

import numpy as np
import scipy as sp
import sympy as sy
import networkx as nx

from abelfunctions.differentials import differentials
from abelfunctions.monodromy     import Monodromy
import abelfunctions.homology    import *

_Z = sympy.Dummy('Z')



def path_to_base_point(self, rs, P):
    """
    Determines a path from P to the base point of the Riemann surface,
    'rs'.
    """


class RiemannSurface_Path():
    """
    Defines a path on the Riemann Surface.
    """
    def __init__(self, Riemann_surface):
        self.Riemann_surface = Riemann_surface
        self.points = []

    def __add__(self, other):
        if isinstance(other, RiemannSurface_Path):
            if other.rs == self.rs:
                points = self.points + other.points
            else:
                raise ValueError("%s is not a path on %s."%(other,self.rs))

    def create_path(self, P1, P2):
        """
        Creates a path from P1 on the Riemann surface to P1.
        """
        pass
        


class RiemannSurface_Point():
    """
    Defines a point on the Riemann surface.
    """
    
    def __init__(self, rs, point):
        """
        Construct a point `(x0,y0)` defined on the Riemann Surface
        `riemann_surface`.
        """
        # efficiently compute points from other Riemann surface points
        if isinstance(point,list) or isinstance(point,tuple):
            x0, y0 = point
            self.rs = rs
            self.x0 = x0
            self.y0 = self.matching_lift(x0,y0)
        elif isinstance(point, RiemannSurface_Point):
            if point.rs != rs:
                raise ValueError("Cannot construct point on %s since %s " + /
                                 "is defined on a different Riemann " + /
                                 "surface." %(rs, point))
            else:
                x0, y0 = point.x0, point.y0
                self.rs = rs
                self.x0 = x0
                self.y0 = y0


    def __repr__(self):
        return "(%s,%s)"%(x0,y0)


    def __add__(self, other):
        """
        Returns a multiple of self where self is considered as an
        element of the Divisor group of the Riemann surface.
        """
        return Divisor(self.Riemann_surface, [self, other])

        
    def matching_lift(self, x0, y0):
        """
        Returns the coordinate `y0_exact` which is treated as the
        "exact" point on the Riemann surface assuming the input `y0`
        is an approximation.
        """
        key      = lambda y: abs(y-y0)
        lift_x0  = self.lift(x0)
        y0_exact = min(lift_x0, key=key)

        if self.rs.f.evalf(subs={self.rs.x:x0, self.rs.y:y0_exact}) > 1e-14:
            raise ValueError("(%s,%s) is not a point on %s."%(x0,y0,self.rs))

        return y0_exact


    def lift(self, x0):
        """
        Computes the lift of the point `x0 \in C` in the complex plane
        to the Riemann surface.
        """
        self.rs.lift(x0)

    def riemann_surface(self):
        """
        Returns the Riemann surface that this point is defined on.
        """
        return self.rs




class Divisor():
    """
    Class for defining a divisor on a Riemann surface.
    """
    def __init__(self, rs, points):
        """
        Creates a divisor on the Riemann surface `riemann_surface` using
        the points found in `points`. `points` can be a list of 
        `RiemannSurface_Points` with multiples included or a list of tuples
        of the form `(P,n)` where `n` is the multiplicity of the point `P`.
        """
        self.rs = rs
        self.data = [self.add_point(P) for P in points]

    def add_point(P, n):
        """
        Adds the points n*P to the divisor.
        """
        P = RiemannSurface_point(self.rs, P)
        if P.rs != self.rs:
            raise ValueError("Cannot add points on different " + /
                             "Riemann surfaces.")
        if P in self.data.keys():
            self.data[P] += n
        else:
            self.data[P] = n
        
        
    
        

class RiemannSurface(Monodromy):
    """
    Class for defining a Riemann surface corresponding to a plane
    algebraic curve.
    """
    def __init__(self, f, x, y):
        super.__init__(f,x,y) self.mon = self.monodromy()

        self.f = f
        self.p = sympy.Poly(f,[x,y])
        self.x = x
        self.y = y

    def __repr__(self):
        return "Riemann surface defined by the plane algebraic curve %s."%f


    def lift(self, x0):
        """
        Comptues the lift of the point x0 in the complex plane to the
        Riemann surface.
        """
        f_x0 = self.f.subs({x:1,y:_Z})
        lift_x0 = f_x0.all_roots(multiple=False,radicals=False)
        
        return lift_x0


    def point(self, x0, y0):
        """
        Returns the point on the Riemann surface closest to
        `(x0,y0)`. (In the event that floating point precision is used
        to calculate the coordinates `(x0,y0)`.
        """
        return RiemannSurface_Point(self,x0,y0)
    
    
    def branch_points(self):
        """
        Treating the Riemann surface as a covering of the 
        `\mathbb{C}_x`-plane, return the branch points of the algebraic 
        curve `f = f(x,y)`.
        """
        pass


    def base_point(self):
        """
        Returns a tuple `(a, (y_0, ..., y_{n-1}))` where `a` is the base
        point of the Riemann surface (the x-point at which the `a`- and 
        `b`-cycles begin and end) and `(y_0, ..., y_{n-1})` are the 
        ordered sheets lying above `a`.
        """
        pass
        
        
    def holomorphic_differentials(self):
        """ 
        Returns the basis of holomorphic differentials defined on the
        Riemann surface.
        """
        return differentials(f,x,y)


    def a_cycle(self, i, Npts=32):
        """
        Returns the ith a_cycle as an array of interpolating points
        starting from and ending at the basepoint of the Riemann
        surface.
        """
        mon

    
    def b_cycle(self, i, Npts=32):
        """
        Returns the ith b_cycle as an array of interpolating points
        starting from and ending at the basepoint of the Riemann
        surface.
        """
        pass


    def _get_cycle(self, k, Npts=32):
        """
        Called by `a_cycle()` and `b_cycle()` to obtain the
        corresponding cycle.
        """
        # compute the homology and grab the kth contour as a list of
        # branch points, number of rotations, and direction
        h = homology(f,x,y)
        kappa = h[0]['linearcombination']
        contours = h[0]['cycles']
        contour = contours[k]

        # determine information about the contour: start / stop
        # points, the permutation of the target base point, and the
        # sheet numbers
        n = len(contour)/2
        cycle = contour + [contour[0]]
        for i in xrange(n):
            bpoint     = cycle[2*i+1][0]    # XXX check indexing
            perm       = cycle[2*i+1][1]
            firstsheet = cycle[2*i]
            lastsheet  = cycle[2*i+2]
            perm = reorder_cycle(perm, firstsheet)

            # generate the path points corresponding to this branch
            # point. use the monodromy graph and the conjugates
            
        
        
        
        



    def integrate(self, f, x, y, P1, P0='basepoint'):
        """
        Integrates the function `f = f(x,y)`, defined on the Riemann surface,
        from the point P1 to the point P2.
        """
        pass



    def period_matrix(self):
        """
        Returns the period matrix `\tau = (A \; B)` where `A_{ij}` is
        the integral of the `j`th holomorphic differential basis element
        about the cycle `a_i`. (`B` is defined in the same way about the 
        `b`-cycles.)
        """
        pass
        

        

        
