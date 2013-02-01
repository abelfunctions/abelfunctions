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



def path_to_base_point(self, rs, P):
    """
    Determines a path from P to the base point of the Riemann surface,
    'rs'.
    """
    pass



def interpolating_circle(R, z0, arg, dir, Npts, endpoint=False):
    """
    Returns a list of sympy.mpmath.mpc points of length Npts
    describing a semicircle in the complex plane with center `z0`,
    radius `R`, initial position `arg`, and direction `dir`.
    """
    R, z0, arg, dir = map(sympy.mpmath.mpc, [R,z0,arg,dir])
    exp = sympy.mpmath.exp
    pi = sympy.mpmath.pi
    j = sympy.mpmath.j
    
    t_pts  = sympy.mpmath.linspace(0, 1, Npts, endpoint=endpoint)
    circle = [R*exp(j*(dir*pi*t + arg)) + z0 for t in t_pts]
    return circle



def interpolating_line(z0, z1, Npts, endpoint=False):
    """
    Returns a list of sympy.mpmath.mpc points of length Npts
    describing a line in the complex plane from z0 to z1.
    """
    z0, z1 = map(sympy.mpmath.mpc, [z0,z1])
    
    t_pts = sympy.mpmath.linspace(0, 1, Npts, endpoint=endpoint)
    line = [z0*(1-t) + z1*t for t in t_pts]
    return line



def path_around_branch_point(G, bpt, rot, Npts):
    """
    Returns a list of interpolating x-points starting from the base
    point going around the branch point, "bpt", "rot" number of times.
    The sign of "rot" determines direction.

    Input:
    
    - G: the "monodromy graph", as computed by Monodromy
    
    - bpt: the index of the target branch point

    - rot: the rotation number and direction of the path going around
    branch point "bpt".

    Output:

    A list of interpolating x-points.
    """
    # Grab the root node.
    root = G.node[bpt]['root']

    # retreive the vertices between the base point vertex and
    # the target vertex.
    path_vertices = nx.shortest_path(G, source=root, target=bpt)

    # retreive the conjugates. If we pass through a vertex that is
    # a conjugate
    conjugates = G.node[bpt]['conjugates']

    # 1) Compute interpolating points for semi-circle / line pairs
    # leading to the circle encircling the target branch
    # point. (Taking conjugates into account.) Conjugation will
    # indicate whether to pass a vertex on the path towards the target
    # either above or below the vertex.
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
            circ_pts = interpolating_circle(curr_radius, curr_value, arg,
                                            dir, Npts)
            path_points.extend(circ_pts)

        # Add the line to the next discriminant point.
        start = curr_value +  curr_edge_index[0]*curr_radius
        end   = next_value +  curr_edge_index[1]*next_radius
        line_pts = interpolating_line(start, end)
        path_points.extend(line_pts)

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
    circ_pts = interpolating_circle(next_radius, next_value, arg, dir, Npts) +\
        interpolating_circle(next_radius, next_value, 
                             arg+sympy.mpmath.pi, dir, Npts)
    circ_pts = circ_pts * int(abs(rot))
    circ_pts.append(circ_pts[0])

    # 3) Combine the path to the circle, the final circle points, and
    # the reverse path points
    path_points.extend(circ_pts + path_points[::-1])2

    return path_points



def compute_lift_points(RS, x_points, y0):
    """
    Given a set of points in the complex x-plane, `x_points`, and a
    starting sheet index, `sheet_index`, on the Riemann surface, `RS`
    compute the set of complex y-points lying over the x-points on the
    Riemann surface.

    Input:

    - `RS`: a RiemannSurface defined by a plane algebraic curve `f =
      f(x,y)`

    - `x_points`: a list of points in the complex x-plane that
    interpolate a path on the cut Riemann surface, RS.

    - `y0`: the y-point at which the path begins


    Output:

    - `y_points`: a list of complex y-points such that (x_points[i],
    y_points[i]) defines an interpolating path on the Riemann surface
    RS.
    """
    n = sympy.degree(RS.f,y)
    eps = sympy.mpmath.eps

    # Create the necessary functions for computing on the Riemann surface
    f = sympy.lambdify([x, y], RS.f, "mpmath")
    _dfdx = sympy.diff(f, x).simplify()
    _dfdy = sympy.diff(f, y).simplify()
    dfdx = sympy.lambdify([x, y], _dfdx, "mpmath")
    dfdy = sympy.lambdify([x, y], _dfdy, "mpmath")

    base_point = RS.base_points()
    
    # Check if (x_points[0], y0) is a point on the Riemann
    # surface. (Or, is at least close enough to the Riemann surface
    # for now.)
    #
    # XXX This function currently only works for paths starting at the
    # base point on the Riemann surface
    if sympy.mpmath.abs(x_points[0] - base_point) > eps / 2.0:
        raise NotImplementedError("Cannot compute arbitrary paths on" + \
                                      " Riemann surfaces, only on"    + \
                                      " those starting at the base point.")

    # Check if y0 is close enough to the Riemann surface. Compute the
    # nearest y_point in order to get closer.
    if sympy.mpmath.abs(f(x_points[0], y0)) < 2.0 * eps:
        raise ValueError("(x_points[0], y0) is not a point on the" + \
                             " Riemann surface.")

    # Initialize analytic continuation loop: get as close as possible
    # to the Riemann surface
    xim1     = x_points[0]
    f_xim1  = lambda y: f(xim1,y)
    df_xim1 = lambda y: - dx * dfdx(xim1,y) / dfdy(xim1,y)
    guess = (y0-eps, y0, y0+eps)
    yim1 = sympy.mpmath.findroot(f_xim1, guess, df=df_xim1, solver='muller')

    # analytic continuation loop
    y_points    = range(len(x_points))
    y_points[0] = yim1
    idx = 1
    max_dx = 0
    for xi in x_points[1:]:
        # compute numerical approximation of the next set of roots
        # using Taylor series. This allows us to use fewer
        # interpolating points. Use generators for fast creation.
        dx = xi - xim1

        f_xi  = lambda y: f(xi,y)
        df_xi = lambda y: - dx * dfdx(xi,y) / dfdy(xi,y)

        yp        = - dfdx(xim1,yim1) / dfdy(xim1,yim1)
        dy        = yp * dx + sympy.mpmath.eps       # in case yp == 0
        yi_approx = yim1 + dy
        guess     = (yi_approx - dy, yi_approx, yi_approx + dy)
        yi        = sympy.mpmath.findroot(f_xi, guess, df=df_xi,
                                          solver='muller')

        # store and update (making the array ahead of time is faster)
        y_points[idx] = yi 
        yim1 = yi
        xim1 = xi
        idx += 1
    
    return y_points
                



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
        self._monodromy = self.monodromy()
        self._homology  = homology(self._monodromy.hurwitz_system())
        

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


    def a_cycle(self, i, Npts=8):
        """
        Returns the ith a_cycle as an array of interpolating points
        starting from and ending at the basepoint of the Riemann
        surface.
        """
        mon

    
    def b_cycle(self, i, Npts=8):
        """
        Returns the ith b_cycle as an array of interpolating points
        starting from and ending at the basepoint of the Riemann
        surface.
        """
        pass


    @cached_function
    def c_cycle(self, i, Npts=8):
        """
        Returns a path in the complex x-plane of the x-points in the
        monodromy path.
        
        Input:

        - i: the index of the c-cycle
        
        - Npts: the number of interpolating points per "path
        segment". A path segment refers to either a line segment
        connecting two monodromy path circles or a semicircle as part
        of a monodromy path circle.

        Output:

        - a list of interpolating x-point of the i'th c-cycle
        """
        # get the cycle data from homology: each c-cycle is a list of
        # alternating sheet numbers s_k and branch point / number of
        # rotations tuples (b_{i_k}, n_k)
        cycles  = self._homology['cycles']
        
        G = self.mondoromy_graph()
        root = G.node[0]['root']
        path_points = []

        # For each (branch point, rotation number) pair appearing in
        # the cycle compute the interpolating points in the complex
        # x-plane going around the given branch points a number of
        # times equal to the rotation number. Add these points to the
        # list of path points.
        for (bpt, rot) in cycles[i][1::2]:
            bpt_path_points = path_around_branch_pont(G, bpt, rot, Npts)
            path_points.extend(bpt_path_points)

        return


    def integrate_c_cycle(self, h, x, y, i, Npts=8)
        """
        Integrates the differential `omega = h(x,y) dx`, defined on the
        Riemann surface, on the ith c-cycle.
        
        Input:

        - `h,x,y`: a Sympy funciton in the variables `x` and `y`

        - `i`: the index of the c-cycle to integrate over.

        - `Npts`: the number of interpolating points per "path
        segment". A path segment refers to either a line segment
        connecting two monodromy path circles or a semicircle as part
        of a monodromy path circle.
        """
        base_lift = self.base_lift()
        y0 = base_lift[0]

        # Determine the x- and y-points defining the path on the
        # Riemann surface.
        x_cycle_points = self.c_cycle(i,Npts=Npts)
        y_cycle_points = compute_lift_points(self, x_cycle_points, y0)

        # Compute h(x,y) over all (x,y) points on the Riemann surface
        h = sympy.lambdify([x,y], h)
        h_points = map(h, zip(x_cycle_points, y_cycle_points))
        
        # Numerically integrate
        #
        # XXX for now, use own trapezoidal rule
        
        

        



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
        
        

        

        
