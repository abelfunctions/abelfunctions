"""
Riemann Surfaces
"""

import numpy as np
import scipy as sp
import sympy as sy
import networkx as nx

from abelfunctions.differentials import differentials
from abelfunctions.monodromy     import Monodromy
import abelfunctions.homology    import homology

_Z = sympy.Dummy('Z')
_t = sympy.Dummy('t')



def path_around_branch_point(G, bpt, rot, Npts):
    """
    Returns a list of sympy functions parameterizing the path starting
    from the base point going around the branch point, "bpt", "rot"
    number of times.  The sign of "rot" determines direction.

    Input:
    
    - G: the "monodromy graph", as computed by Monodromy
    
    - bpt: the index of the target branch point

    - rot: the rotation number and direction of the path going around
    branch point "bpt".

    Output:

    A list of path segments defined by sympy functions.
    """
    # Grab the root node.
    root = G.node[bpt]['root']

    # retreive the vertices between the base point vertex and
    # the target vertex.
    path_vertices = nx.shortest_path(G, source=root, target=bpt)

    # retreive the conjugates. If we pass through a vertex that is
    # a conjugate
    conjugates = G.node[bpt]['conjugates']

    # 1) Compute path data for semi-circle / line pairs leading to the
    # circle encircling the target branch point. (Taking conjugates
    # into account.) Conjugation will indicate whether to pass a
    # vertex on the path towards the target either above or below the
    # vertex.
    path_data = []
    prev_node = root
    for idx in range(len(path_vertices)-1):
        curr_node   = path_vertices[idx]
        curr_radius = G.node[curr_node]['radius']
        curr_value  = G.node[curr_node]['value']

        next_node   = path_vertices[idx+1]
        next_radius = G.node[next_node]['radius']
        next_value  = G.node[next_node]['value']

        # Determine if semi-circle is needed. This is done by checking
        # the path index of the previous edge with the path index of
        # the next edge. If needed, add semicircle going in the
        # appropriate direction where the direction is determined by
        # the conjugation list. A special case is taken if we're 
        # at the root vertex.
        curr_edge_index = G[curr_node][next_node]['index']
        if prev_node == root:
            prev_edge_index = (0,-1)  # in the root node case
        else:
            prev_edge_index = G[prev_node][curr_node]['index']
        if prev_edge_index[1] != curr_edge_index[0]:
            arg = sympy.mpmath.pi if prev_edge_index[1] == -1 else 0
            dir = -1 if curr_node in conjugates else 1
            path_data.append((curr_radius, curr_value, arg, dir))

        # Add the line to the next discriminant point.
        start = curr_value +  curr_edge_index[0]*curr_radius
        end   = next_value +  curr_edge_index[1]*next_radius
        path_data.append((start,end))

        # Update previous point
        prev_node = curr_node


    # 2) Construct interpolating points around the target
    # branch point. The rotation number "rot" tells us how
    # many times to go around the branch point and in which
    # direction. There's a special case for when we just
    # encircle the root node.
    if len(path_vertices) == 1:
        next_value  = G.node[root]['value']
        next_radius = G.node[root]['radius']
        curr_edge_index = (-1,-1)

    arg = sympy.mpmath.pi if curr_edge_index[1] == -1 else 0
    dir = 1 if rot > 0 else -1
    circle_data = [
        (next_radius, next_value, arg, dir)
        (next_radius, next_value, arg+sympy.mpmath.pi, dir)
        ]
    circle_data = circle_data * int(abs(rot))

    # 3) Use the path data to compute parameterizations of each of the
    # path segments.
    path_segments = []
    exp = sympy.mpmath.exp
    pi = sympy.mpmath.pi
    j = sympy.mpmath.j

    # add the segments leading up to the branch point circle
    for datum in path_data:
        if len(datum) == 2:
            z0,z1     = map(sympy.mpmath.mpc,datum)
            segment   = lambda t: z0*(1-t) + z1*t
            d_segment = lambda t: -z0 + z1
        else:
            R,w,arg,dir = map(sympy.mpmath.mpc,datum)
            segment     = lambda t: R*exp(j*(dir*pi*t + arg)) + w
            d_segment   = lambda t: R*j*dir*pi * exp(j*(dir*pi*t + arg))

        path_segments.append( (segment, d_segment) )

    # add the semicircle segments going around the branch point
    for datum in circle_data:
        R,w,arg,dir = map(sympy.mpmath.mpc,datum)
        segment     = lambda t: R*exp(j*(dir*pi*t + arg)) + w
        d_segment   = lambda t: R*j*dir*pi * exp(j*(dir*pi*t + arg))
        path_segments.append( (segment, d_segment) )

    # add the reversed path segments leading back to the base point
    # (to reverse these paths, simply make the transformation t |-->
    # (1-t) )
    for datum in reversed(path_data):
        if len(datum) == 2:
            z0,z1     = map(sympy.mpmath.mpc,datum)
            segment   = lambda t: z0*t + z1*(1-t)
            d_segment = lambda t: z0 - z1
        else:
            R,w,arg,dir = map(sympy.mpmath.mpc,datum)
            segment     = lambda t: R*exp(j*(dir*pi*(1-t) + arg)) + w
            d_segment   = lambda t: -R*j*dir*pi * exp(j*(dir*pi*(1-t) + arg))

        path_segments.append( (segment, d_segment) )
    
    return path_segments



# def analytically_continue(RS, path, point):
#     """
#     Given a parametrized path `path` in the complex x-plane and a
#     Riemann surface, `RS`, analytically continue the point
#     `point=(x0,y0)` along the path segemnt.

#     Input:

#     - `RS`: a Riemann surface

#     - `path`: a parameterized segment of the projection of the path to
#       the complex x-plane. Specifically, a continuous map t |--> x(t).

#     - `point`: a pont (x0,y0) on the Riemann surface to analytically
#     continue along the path


#     Output:

#     - `(xn,yn)`: the point at the end of the path.
#     """
#     n = sympy.degree(RS.f,y)
#     eps = sympy.mpmath.eps
#     x0, y0 = map(sympy.mpmath.mpc,point)

#     # Create the necessary functions for computing on the Riemann surface
#     f = sympy.lambdify([x, y], RS.f, "mpmath")
#     _dfdx = sympy.diff(f, x).simplify()
#     _dfdy = sympy.diff(f, y).simplify()
#     dfdx = sympy.lambdify([x, y], _dfdx, "mpmath")
#     dfdy = sympy.lambdify([x, y], _dfdy, "mpmath")

#     base_point = RS.base_point()
    
#     # Check if (x0, y0) is a point on the Riemann surface (or, is at
#     # least close enough to the Riemann surface for now.) and is at the
#     # start of the x-path.
#     if sympy.mpmath.abs(x0 - path(0)) > eps / 2.0:
#         raise ValueError("Point is not at the start of the path.")

#     # Check if y0 is close enough to the Riemann surface. Compute the
#     # nearest y_point in order to get closer.
#     if sympy.mpmath.abs(f(x0, y0)) < 2.0 * eps:
#         raise ValueError("`point` is not a point on the Riemann surface.")

#     # Initialize analytic continuation loop: get as close as possible
#     # to the Riemann surface
#     xim1     = x0
#     f_xim1  = lambda y: f(xim1,y)
#     df_xim1 = lambda y: - dx * dfdx(xim1,y) / dfdy(xim1,y)
#     guess = (y0-eps, y0, y0+eps)
#     yim1 = sympy.mpmath.findroot(f_xim1, guess, df=df_xim1, solver='muller')

#     # analytic continuation loop
#     y_points    = range(len(x_points))
#     y_points[0] = yim1
#     idx = 1
#     max_dx = 0
#     while abs(t-1) < eps:
#         # compute numerical approximation of the next set of roots
#         # using Taylor series. This allows us to use fewer
#         # interpolating points. Use generators for fast creation.
#         dx = xi - xim1

#         f_xi  = lambda y: f(xi,y)
#         df_xi = lambda y: - dx * dfdx(xi,y) / dfdy(xi,y)

#         yp        = - dfdx(xim1,yim1) / dfdy(xim1,yim1)
#         dy        = yp * dx + sympy.mpmath.eps       # in case yp == 0
#         yi_approx = yim1 + dy
#         guess     = (yi_approx - dy, yi_approx, yi_approx + dy)
#         yi        = sympy.mpmath.findroot(f_xi, guess, df=df_xi,
#                                           solver='muller')

#         # store and update (making the array ahead of time is faster)
#         y_points[idx] = yi 
#         yim1 = yi
#         xim1 = xi
#         idx += 1
    
#     return y_points
                



# class RiemannSurface_Path():
#     """
#     Defines a path on the Riemann Surface.

#     Input:

#     - Riemann_
#     """
#     def __init__(self, RS, x_path_segments=[]):
#         self.RS = RS
#         self.x_path_segments = x_path_segments
#         self.N_segments = len(x_path_segments)

#     def __add__(self, other):
#         if isinstance(other, RiemannSurface_Path):
#             if other.RS == self.RS:
#                 self.x_path_segments.extend(other.x_path_segments)
#             else:
#                 raise ValueError("%s is not a path on %s."%(other,self.RS))

#     def create_path(self, P1, P2):
#         """
#         Creates a path from P1 on the Riemann surface to P1.
#         """
#         pass
        


# class RiemannSurface_Point():
#     """
#     Defines a point on the Riemann surface.
#     """
    
#     def __init__(self, rs, point):
#         """
#         Construct a point `(x0,y0)` defined on the Riemann Surface
#         `riemann_surface`.
#         """
#         # efficiently compute points from other Riemann surface points
#         if isinstance(point,list) or isinstance(point,tuple):
#             x0, y0 = point
#             self.rs = rs
#             self.x0 = x0
#             self.y0 = self.matching_lift(x0,y0)
#         elif isinstance(point, RiemannSurface_Point):
#             if point.rs != rs:
#                 raise ValueError("Cannot construct point on %s since %s " + /
#                                  "is defined on a different Riemann " + /
#                                  "surface." %(rs, point))
#             else:
#                 x0, y0 = point.x0, point.y0
#                 self.rs = rs
#                 self.x0 = x0
#                 self.y0 = y0


#     def __repr__(self):
#         return "(%s,%s)"%(x0,y0)


#     def __add__(self, other):
#         """
#         Returns a multiple of self where self is considered as an
#         element of the Divisor group of the Riemann surface.
#         """
#         return Divisor(self.Riemann_surface, [self, other])

        
#     def matching_lift(self, x0, y0):
#         """
#         Returns the coordinate `y0_exact` which is treated as the
#         "exact" point on the Riemann surface assuming the input `y0`
#         is an approximation.
#         """
#         key      = lambda y: abs(y-y0)
#         lift_x0  = self.lift(x0)
#         y0_exact = min(lift_x0, key=key)

#         if self.rs.f.evalf(subs={self.rs.x:x0, self.rs.y:y0_exact}) > 1e-14:
#             raise ValueError("(%s,%s) is not a point on %s."%(x0,y0,self.rs))

#         return y0_exact


#     def lift(self, x0):
#         """
#         Computes the lift of the point `x0 \in C` in the complex plane
#         to the Riemann surface.
#         """
#         self.rs.lift(x0)

#     def riemann_surface(self):
#         """
#         Returns the Riemann surface that this point is defined on.
#         """
#         return self.rs




# class Divisor():
#     """
#     Class for defining a divisor on a Riemann surface.
#     """
#     def __init__(self, rs, points):
#         """
#         Creates a divisor on the Riemann surface `riemann_surface` using
#         the points found in `points`. `points` can be a list of 
#         `RiemannSurface_Points` with multiples included or a list of tuples
#         of the form `(P,n)` where `n` is the multiplicity of the point `P`.
#         """
#         self.rs = rs
#         self.data = [self.add_point(P) for P in points]

#     def add_point(P, n):
#         """
#         Adds the points n*P to the divisor.
#         """
#         P = RiemannSurface_point(self.rs, P)
#         if P.rs != self.rs:
#             raise ValueError("Cannot add points on different " + /
#                              "Riemann surfaces.")
#         if P in self.data.keys():
#             self.data[P] += n
#         else:
#             self.data[P] = n
        
        
    
        

class RiemannSurface(Monodromy):
    """
    Class for defining a Riemann surface corresponding to a plane
    algebraic curve.
    """
    def __init__(self, f, x, y):
        super.__init__(f,x,y) self.mon = self.monodromy()
        self._monodromy = self.monodromy()
        self._homology  = homology(self._monodromy.hurwitz_system())

        _dfdx = sympy.diff(f, x).simplify()
        _dfdy = sympy.diff(f, y).simplify()

        self.p    = sympy.Poly(f,[x,y])
        self.f    = sympy.lambdify([x,y], f, "mpmath")
        self.dfdx = sympy.lambdify([x, y], _dfdx, "mpmath")
        self.dfdy = sympy.lambdify([x, y], _dfdy, "mpmath")
        self.x = x
        self.y = y
        
        self.n = sympy.degree(f,y)



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
        return super.branch_points()



    def base_point(self):
        """
        Returns a tuple `(a, (y_0, ..., y_{n-1}))` where `a` is the base
        point of the Riemann surface (the x-point at which the `a`- and 
        `b`-cycles begin and end) and `(y_0, ..., y_{n-1})` are the 
        ordered sheets lying above `a`.
        """
        return super.base_point()

        
        
    def holomorphic_differentials(self):
        """ 
        Returns the basis of holomorphic differentials defined on the
        Riemann surface.
        """
        return differentials(f,x,y)



    def analytically_continue(self, path, y0, tend):
        """
        Given a parametrized path `path` in the complex x-plane and a
        Riemann surface, `RS`, analytically continue the point
        `point=(x0,y0)` along the path segemnt.

        Input:

        - `path`: a parameterized segment of the projection of the path to
          the complex x-plane. Specifically, a continuous map t |--> x(t).

        - `y0`: a point on the complex y-plane lying above the start of `path`
        such that (path(t=0), y0) is a point on the Riemann surface


        Output:

        - `yn`: the complex y-point at the end of the path.
        """
        eps = sympy.mpmath.eps

        x0 = sympy.mpmath.mpmathify(path[0](0))
        y0 = sympy.mpmath.mpmathify(y0)
        base_point = self.base_point()

        # Check if y0 is close enough to the Riemann surface. Compute the
        # nearest y_point in order to get closer.
        if sympy.mpmath.abs(f(x0, y0)) < 2.0 * eps:
            raise ValueError("`point` is not a point on the Riemann surface.")

        # Initialize analytic continuation loop: get as close as possible
        # to the Riemann surface
        xim1    = x0
        f_xim1  = lambda y: f(xim1,y)
        df_xim1 = lambda y: - dx * dfdx(xim1,y) / dfdy(xim1,y)
        guess = (y0-eps, y0, y0+eps)
        yim1 = sympy.mpmath.findroot(f_xim1, guess, df=df_xim1, solver='muller')

        # Analytic continuation loop: start with dt = 1. Take a step
        # using Taylor series. Use a root finder to get to the closest
        # point on the Riemann surface. Check if this point is on the
        # correct sheet. If not, have the dt and try again
        tim1 = 0
        ti   = tend
        xi   = path[0](ti)
        while tim1 != tend:
            dx   = xi-xim1
            f_xi  = lambda y: f(xi,y)
            dy_xi = lambda y: - dx * dfdx(xi,y) / dfdy(xi,y)

            # Take a continuation step using a first order Taylor
            # series.  (The root finder needs a 3-tuple guess so we
            # add a little bit of error to dy.)
            yp        = - dfdx(xim1,yim1) / dfdy(xim1,yim1)
            dy        = yp * dx + sympy.mpmath.eps
            yi_approx = yim1 + dy
            guess     = (yi_approx - dy, yi_approx, yi_approx + dy)
            yi        = sympy.mpmath.findroot(f_xi, guess, df=df_xi,
                                              solver='muller')

            # Check if we're on the correct branch: yi (the point on
            # the RS) should be closer to yi_approx (the Taylor approx
            # of yi) than to yim1 (the previous point). If we're
            # further away then the root finder had trouble.
            if sympy.mpmath.abs(yi-yi_approx) < sympy.mpmath.abs(yi-yim1):
                tim1 = ti
                ti = tend
                xim1 = xi
                xi = path[0](ti)
            else:
                ti = (ti + tim1) / 2.0
                xi = path[0](ti)

        return yi



    def a_cycle(self, i, Npts=8):
        """
        Returns the ith a_cycle as an array of interpolating points
        starting from and ending at the basepoint of the Riemann
        surface.
        """
        pass

    
    def b_cycle(self, i, Npts=8):
        """
        Returns the ith b_cycle as an array of interpolating points
        starting from and ending at the basepoint of the Riemann
        surface.
        """
        pass



    def _c_cycle_segments(self, i):
        # get the cycle data from homology: each c-cycle is a list of
        # alternating sheet numbers s_k and branch point / number of
        # rotations tuples (b_{i_k}, n_k)
        cycles  = self._homology['cycles']
        
        G = self.mondoromy_graph()
        root = G.node[0]['root']
        path_segments = []

        # For each (branch point, rotation number) pair appearing in
        # the cycle compute the interpolating points in the complex
        # x-plane going around the given branch points a number of
        # times equal to the rotation number. Add these points to the
        # list of path points.
        for (bpt, rot) in cycles[i][1::2]:
            bpt_path_segments = path_around_branch_pont(G, bpt, rot, Npts)
            path_segments.extend(bpt_path_segments)

        return path_segments


#     def c_cycle(self, i):
#         """
#         Returns a path in the complex x-plane of the x-points in the
#         monodromy path.
        
#         Input:

#         - i: the index of the c-cycle
        
#         - Npts: the number of interpolating points per "path
#         segment". A path segment refers to either a line segment
#         connecting two monodromy path circles or a semicircle as part
#         of a monodromy path circle.

#         Output:

#         - a list of interpolating x-point of the i'th c-cycle
#         """
#         # get the cycle data from homology: each c-cycle is a list of
#         # alternating sheet numbers s_k and branch point / number of
#         # rotations tuples (b_{i_k}, n_k)
#         cycles  = self._homology['cycles']
        
#         G = self.mondoromy_graph()
#         root = G.node[0]['root']
#         path_points = []

#         # For each (branch point, rotation number) pair appearing in
#         # the cycle compute the interpolating points in the complex
#         # x-plane going around the given branch points a number of
#         # times equal to the rotation number. Add these points to the
#         # list of path points.
#         for (bpt, rot) in cycles[i][1::2]:
#             bpt_path_points = path_around_branch_pont(G, bpt, rot, Npts)
#             path_points.extend(bpt_path_points)

#         return None


    def integrate_c_cycle(self, omega, x, y, i)
        """
        Integrates the differential `omega`, defined on the
        Riemann surface, on the ith c-cycle with respect to dx.
        
        Input:

        - `omega,x,y`: a Sympy funciton in the variables `x` and `y`

        - `i`: the index of the c-cycle to integrate over.
        """
        base_lift = self.base_lift()
        y0 = base_lift[0]

        # compute the cycle path segments
        path_segments = self._c_cycle_segments(i)
        omega = sympy.lambdify([x,y], omega)
        
        # Numerically integrate over each path segment
        integral = sympy.mpmath.mpc(0)
        quad = sympy.mpmath.quadgl
        for X, dXdt in path_segments:
            Y = lambda t: self.analytically_continue((X,dXdt), y0, t)
            g = lambda t: omega(X(t), Y(t)) * dXdt(t) 
            
            integral += quad(g,[0,1])
            y0 = self.analytically_continue((X,dXdt), y0, 1)

        return integral



    def period_matrix(self):
        """
        Returns the period matrix `\tau = (A \; B)` where `A_{ij}` is
        the integral of the `j`th holomorphic differential basis element
        about the cycle `a_i`. (`B` is defined in the same way about the 
        `b`-cycles.)

        Note: this function computes the integrals of the
        differentials over each of the c-cycles and then uses the
        homology data to determine which linear combination of
        c-cycles integrals gives the a- and b-cycles integrals.
        """
        
        

        

        
