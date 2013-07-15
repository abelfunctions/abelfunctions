"""
Riemann Surface Paths

This module defines paths on Riemann surfaces, codified in the class
`RiemannSurface_Path`. The primary components of a `RiemannSurface_Path`
are

- a path, as a piecewise collection of semi-circles and lines, in the
complex x-plane.

- a mechanisim for analytically continuing any or all y-roots lying
  above this complex x-plane path
"""

import numpy
import scipy
import sympy
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

import pdb

def factorial(n):
    return reduce(lambda a,b: a*b, xrange(1,n+1))

def newton(df,xip1,yij):
    step = 1
    while numpy.abs(step) > 1e-15:
        # check if Df is invertible. (If not, then we are at a
        # critical point.)
        df1 = df[1](xip1,yij)
        if numpy.abs(df1) < 1e-15:
            return yij

        # Newton iterate
        step = df[0](xip1,yij)/df1
        yij -= step

    return yij


def smale_beta(df,xip1,yij):
    return numpy.abs( df[0](xip1,yij)/df[1](xip1,yij) )


def smale_gamma(df,xip1,yij,deg):
    df1 = df[1](xip1,yij)
    bounds = ( numpy.abs(df[k](xip1,yij)/(factorial(k)*df1))**(1./(k-1.0))
               for k in xrange(2,deg+1) )
    return max(bounds)


def smale_alpha(df,xip1,yij,deg):
    return smale_beta(df,xip1,yij) * smale_gamma(df,xip1,yij,deg)


def polyroots(f,x,y,xi,types='numpy'):
    """
    Helper function for computing multiprecise roots of polynomials
    using `sympy.mpmath`.

    Precision is set by modifying `sympy.mpmath.mp.dps`.

    Input:

    - `f,x,y`: a complex plane algebraic curve in x,y with
      `sympy.mpmath.mpc` coefficients

    - `xi`: a complex x-point

    Output:

    - the multiprecise roots of ``f(xi,y) = 0``.
    """
    dps = sympy.mpmath.mp.dps
    p = f.as_poly(y)

    if types == 'mpmath':
        coeffs = [c.evalf(subs={x:xi},n=dps) for c in p.all_coeffs()]
        coeffs = [sympy.mpmath.mpc(*(z.as_real_imag())) for z in coeffs]
        return sympy.mpmath.polyroots(coeffs)
    else:
        coeffs = [c.evalf(subs={x:xi},n=15) for c in p.all_coeffs()]
        coeffs = [numpy.complex(z) for z in coeffs]
        poly = numpy.poly1d(coeffs)
        return (poly.r).tolist()

    return sympy.mpmath.polyroots(coeffs)


def _line_path(a=None,b=None):
    """
    Returns a function-derivative pair (x(t), x'(t)) parameterizing a straight
    line between the complex numbers a and b for `t \in [0,1]`.
    """
    return (
        lambda t: a*(1-t) + b*t,
        lambda t: b-a
        )

def _circle_path(R=None,w=None,arg=None,dir=None,
                 exp=None,I=None,PI=None,rev=False):
    """
    Returns a function-derivative pair (x(t),x'(t)) parameterizing a semicircle
    in the complex plane with center w, radius R, starting angle arg, and
    direction dir.

    Additional arguments determine which functions to use and whether or not to
    compute the same path but in the reverse direction.
    """
    if rev:
        return (
            lambda t: R * exp(I*(dir*PI*(1-t) + arg)) + w,
            lambda t: -(R*I*PI*dir) * exp(I*(dir*PI*(1-t) + arg))
            )
    else:
        return (
            lambda t: R * exp(I*(dir*PI*t + arg)) + w,
            lambda t: (R*I*PI*dir) * exp(I*(dir*PI*t + arg))
            )

def _path_segments_from_path_data(path_data, circle_data, types='numpy'):
    """
    Take data about the x-path and returns parameterizing functions.

    Input:

    - `path_data`: a list containing tuples

        (z0,z1)

        or

        (R,w,arg,d)

        containing information about how to get to the final circle of
        the x-path.

    - `circle_data`: a list containing tuples (R,w,arg,d) describing
      how to go around the circle.

    Output:

    [x(t),dxdt(t)]
    """
    path_segments = []
    if types == 'mpmath':
        exp = sympy.mpmath.exp
        pi = sympy.mpmath.pi
        j = sympy.mpmath.j
        cast_type = sympy.mpmath.mpc
    else:
        exp = numpy.exp
        pi = numpy.pi
        j = numpy.complex(1.0j)
        cast_type = numpy.complex

    # Add the segments leading up to the branch point circle
    for datum in path_data:
        if len(datum) == 2:
            z0,z1 = map(cast_type,datum)
            path_part = _line_path(a=z0,b=z1)
        else:
            R,w,arg,dir = map(cast_type,datum)
            path_part = _circle_path(R=R,w=w,arg=arg,dir=dir,exp=exp,
                                     I=j,PI=pi,rev=False)
        path_segments.append(path_part)

    # Add the semicircle segments going around the branch point
    for datum in circle_data:
        R,w,arg,dir = map(cast_type,datum)
        path_part = _circle_path(R=R,w=w,arg=arg,dir=dir,exp=exp,I=j,PI=pi)
        path_segments.append(path_part)

    # Add the reversed path segments leading back to the base point
    # (to reverse these paths, simply make the transformation t |-->
    # (1-t) )
    for datum in reversed(path_data):
        if len(datum) == 2:
            z0,z1 = map(cast_type,datum)
            path_part = _line_path(a=z1,b=z0)
        else:
            R,w,arg,dir = map(cast_type,datum)
            path_part = _circle_path(R=R,w=w,arg=arg,dir=dir,exp=exp,
                                     I=j,PI=pi,rev=True)
        path_segments.append(path_part)

    return path_segments


def path_around_branch_point(G, bpt, rot, types='numpy'):
    """
    Returns a list of lambda functions parameterizing the path starting
    from the base point going around the branch point, "bpt", "rot"
    number of times.  The sign of "rot" determines direction.

    Input:

    - G: the "monodromy graph", as computed by monodromy_graph()

    - bpt: the index of the target branch point

    - rot: the rotation number and direction of the path going around
    branch point "bpt".

    Output:

    A list of path segments ``(x(t), dxdt(t))`` for ``t \in [0,1]``
    defined by lambda functions.
    """
    if types == 'mpmath':
        abs = sympy.mpmath.abs
        pi = sympy.mpmath.pi
    else:
        abs = numpy.abs
        pi = numpy.pi

    ## HACK
    root = G.node[0]['root']

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
            arg = pi if prev_edge_index[1] == -1 else 0
            dir = -1 if prev_node in conjugates else 1 # XXX
            path_data.append((curr_radius, curr_value, arg, dir))

        # Add the line to the next discriminant point.
        start = curr_value +  curr_edge_index[0]*curr_radius
        end   = next_value +  curr_edge_index[1]*next_radius
        path_data.append((start,end))

        # Update previous point
        prev_node = curr_node

    # Construct interpolating points around the target
    # branch point. The rotation number "rot" tells us how
    # many times to go around the branch point and in which
    # direction. There's a special case for when we just
    # encircle the root node.
    if len(path_vertices) == 1:
        next_value  = G.node[root]['value']
        next_radius = G.node[root]['radius']
        curr_edge_index = (-1,-1)

    arg = pi if curr_edge_index[1] == -1 else 0
    dir = 1 if rot > 0 else -1
    circle_data = [
        (next_radius, next_value, arg, dir),
        (next_radius, next_value, arg+pi, dir)
        ]
    circle_data = circle_data * int(abs(rot))

    # turn this data into parameterized path segments (with derivatives)
    path_segments = _path_segments_from_path_data(path_data,circle_data,
                                                  types=types)
    return path_segments


def path_around_infinity(G, rot, types='numpy'):
    """
    Returns a list of labmda functions paramterizing the path starting
    from the base point and going around infinity.

    Input:

    - G: the "monodromy graph", as computed by monodromy_graph()

    - bpt: the index of the target branch point

    - rot: the rotation number and direction of the path going around
    branch point "bpt".

    Output:

    A list of path segments ``(x(t), dxdt(t))`` for ``t \in [0,1]``
    defined by lambda functions.
    """
    if types == 'mpmath':
        abs = sympy.mpmath.absmin
        arg = sympy.mpmath.arg
        pi = sympy.mpmath.pi
        CC = sympy.mpmath.mp.mpc
    else:
        abs = numpy.abs
        arg = numpy.angle
        pi = numpy.pi
        CC = numpy.complex

    # determine the center of the circle
    values = [data['value'] for node,data in G.nodes(data=True)]
    center = 0 #sum(values)/len(values)

    # the radius of the circle is the distance from the center to the furthest
    # away branch point plus the monodromy path radius at that branch point
    radius = 0
    for node,data in G.nodes(data=True):
        node_value = data['value']
        node_radius = data['radius']

        current_radius = abs(node_value) + node_radius
        radius = current_radius if current_radius > radius else radius
        if current_radius > radius:
            radius = current_radius
        max(abs(value-center) for value in values)

    # the base point is chosen to be furthest to the left of all branch points.
    # travel along the line made by the base point and the center until we
    # reach the perimeter of the circle.
    base_point = CC(G.node[0]['basepoint'])
    z0 = CC(base_point)
    arg0 = arg(z0)               # starting angle on the circle
    z1 = CC((z0/abs(z0))*radius) # starting point on the circle
    dir = 1 if rot > 0 else -1

    # construct path
    if abs(base_point - z1) > 10**(-14):
        path_data = [(base_point,z1)]
    else:
        path_data = []

    circle_data = [(radius,center,arg0,dir), (radius,center,arg0+pi,dir)]
    circle_data = circle_data * int(abs(rot))

    # turn this data into parameterized path segments (with derivatives)
    path_segments = _path_segments_from_path_data(path_data,circle_data,
                                                  types=types)
    return path_segments


class RiemannSurfacePath():
    """
    Defines a path on a Riemann surface parameterized on the interval
    [0,1]. Used for analytically continuing and integrating on Riemann
    surfaces.
    """
    def __init__(self, RS, P0, path_segments=None, types='numpy'):
        """
        Create a path on the RiemannSurface, `RS`, starting at the
        RiemannSurfacePoint, `P0`.

        Input:

        - `RS`: a RiemannSurface on which the path is defined. (Or, a
          tuple containing `(f,x,y)` where ``f(x,y) = 0`` defined the
          Riemann surface.

        - `P0`: a RiemannSurfacePoint where the path begins. `P0[0]` is the
        base point and `P0[1]` is the base fibre

        - `P1`: (default: `None`) a RiemannSurfacePoint where the path ends.

        - `types`: (default: `'numpy'`) chose which underlying data types to
        use. Options include

          * `'numpy'`: use `numpy.complex` data types

          * `'mpmath'`: use `sympy.mpmath.mp.mpc` data types
        """
        if isinstance(RS,tuple):
            f,x,y = RS
            self.f = f
            self.x = x
            self.y = y
        else:
            self.f = RS.f
            self.x = RS.x
            self.y = RS.y

        # cast into appropriate data type
        if types == 'mpmath':
            CC = sympy.mpmath.mp.mpc
        else:
            CC = numpy.complex

        x0 = CC(P0[0])
        y0 = tuple(map(CC,P0[1]))
        self.P0 = (x0,y0)
        self.deg = sympy.degree(self.f,self.y)
        self.types = types
        self.alpha0 = CC(13.0 - 2.0*numpy.sqrt(17.0))/4.0

        # construct fast, type-cast lambda functions from the algebraic
        # curve. used in performing the Taylor step in the analytic
        # continuation
        self.df = [
            sympy.lambdify((self.x,self.y),sympy.diff(self.f,self.y,k),
                           self.types)
            for k in range(self.deg+1)
            ]

        # path data
        if path_segments == None:
            raise NotImplementedError("Automatic path construction not " + \
                                      "available.")
        self._path_segments     = path_segments
        self._num_path_segments = len(path_segments)

        # initialize the checkpoints: in each path segment analytically
        # continue to a certain number of points along the path and store in
        # memory. This is done so one doesn't have to analytically continue
        # from the base point every time you want to find a point on the path
        self._checkpoints = {}
        for n in range(self._num_path_segments):
            self._checkpoints[n] = []
        self._checkpoints[0].append((0,self.P0))
        self._initialize_checkpoints()


    def __repr__(self):
        return 'Path on the Riemann surface defined by the curve %s = 0'%self.f


    def _initialize_checkpoints(self):
        """
        Analytically continue along the entire path recording the value
        of the roots at equally-spaced "checkpoints" along the path.
        """
        ppseg = 16
        if self.types == 'mpmath':
            t_pts = sympy.mpmath.linspace(0,1,ppseg,endpoint=True)
        else:
            t_pts = numpy.linspace(0,1,ppseg,endpoint=True)

        # for each path segment, analytically continue along the segment and
        # record the checkpoints at equally spaced times.  the first checkpoint
        # of each segment is equal to the last checkpoint of the previous
        # segment.
        for idx in range(self._num_path_segments):
            path_segment = self._path_segments[idx]
            tim1,Pim1 = self._checkpoints[idx][0]

            # analytically continue along segment
            for ti in t_pts[1:]:
                Pi = self.analytically_continue_segment(path_segment,ti,
                                                        checkpoint=(tim1,Pim1))
                self._checkpoints[idx].append((ti,Pi))
                tim1 = ti
                Pim1 = Pi

            # initialize next segment
            if idx != (self._num_path_segments-1):
                self._checkpoints[idx+1].append((0,Pi))


    def _nearest_checkpoint(self, path_segment_index, t):
        """
        Returns the nearest checkpoint to the input, `t`.

        RiemannSurfacePath performs some caching of analytic
        continuation to save on the cost of numerical root finding.
        If `self.__call__()` requires many iterations to compute an
        analytic continuation then checkpoint data is saved in a cache
        making future analytic continuations potentially faster.
        """
        # scan the list of checkpoints for the closest checkpoint.
        # occuring before t.
        tim1 = 0
        Pim1 = self.P0
        for ti, Pi in self._checkpoints[path_segment_index]:
            if ti >= t:
                return (tim1, Pim1)
            else:
                tim1 = ti
                Pim1 = Pi

        # use last point as checkpoint if a successor isn't found
        return ti, Pi



    def __call__(self, t, Npts=8):
        return self.analytically_continue(t,Npts=Npts)


    def step(self, path_segment, ti, tip1, yi):
        """
        An analytic continuation step of the y-roots of f(x,y) from xi to
        xip1.

        In the event that the conditions for Newton-based continuation are
        not satisfied the step function is called recursively to an
        earlier point.

        TODO: write a clean, non-recursive version of this algorithm
        """
        df = self.df
        deg = self.deg
        xi = path_segment[0](ti)
        xip1 = path_segment[0](tip1)

        # raise error if step size is too small
        if numpy.abs(xip1-xi) < 1e-15:
            raise ValueError("Analytic continuation failed.")

        # first determine if the y-root guesses are 'approximate solutions'. if
        # one of them is not then refine the step
        for yij in yi:
            if smale_alpha(df,xip1,yij,deg) > self.alpha0:
                # not an approximate solution. take a half step and return the
                # second half step
                ti_half = (ti+tip1)/2.0
                xi_half,yi_half = self.step(path_segment,ti,ti_half,yi)
                return self.step(path_segment,ti_half,tip1,yi_half)

        # next, determine if the approximate solutions will converge to
        # different associated solutions
        for j in xrange(deg):
            yij = yi[j]
            for k in xrange(j+1,deg):
                yik = yi[k]
                betaij = smale_beta(df,xip1,yij)
                betaik = smale_beta(df,xip1,yik)

                if numpy.abs(yij-yik) < 2*(betaij+betaik):
                    # approximate solutions don't lead to distinct
                    # roots. refine and return
                    ti_half = (ti+tip1)/2.0
                    xi_half,yi_half = self.step(path_segment,ti,ti_half,yi)
                    return self.step(path_segment,ti_half,tip1,yi_half)

        # finally, since we know that we have approximate solutions that will
        # converge to difference associated solutions we will Netwon iterate
        yip1 = [ newton(df,xip1,yij) for yij in yi ]

        # return the t-point that we were able to step to as well as
        # the y-roots lying above that t-point
        return xip1, yip1


#     def step(self, path_segment, ti, tip1, yi):
#         """
#         An analytic continuation step of the y-roots of f(x,y) from xi to
#         xip1.

#         In the event that the conditions for Newton-based continuation are
#         not satisfied the step function is called recursively to an
#         earlier point.

#         TODO: write a clean, non-recursive version of this algorithm
#         """
#         df = self.df
#         deg = self.deg
#         t = tip1

#         while t <= tip1:
#             xi = path_segment[0](ti)
#             x = path_segment[0](t)

#             # raise error if step size is too small
#             if numpy.abs(x-xi) < 1e-15:
#                 raise ValueError("Analytic continuation failed.")

#             # first determine if the y-root guesses are 'approximate
#             # solutions'. if one of them is not then refine the step
#             approximate_solution = True
#             distinct_solutions = True
#             for yij in yi:
#                 if smale_alpha(df,x,yij,deg) > self.alpha0:
#                     # not an approximate solution. refine and return
#                     #return self.step(path_segment,ti,(ti+tip1)/2.0,yi)
#                     t = (ti+t)/2.0
#                     approximate_solution = False
#                     break

#             if approximate_solution:
#                 # next, determine if the approximate solutions will converge to
#                 # different associated solutions
#                 for j in xrange(deg):
#                     yij = yi[j]
#                     betaij = smale_beta(df,x,yij)
#                     for k in xrange(j+1,deg):
#                         yik = yi[k]
#                         betaik = smale_beta(df,x,yik)

#                         if numpy.abs(yij-yik) < 2*(betaij+betaik):
#                             # approximate solutions don't lead to distinct
#                             # roots. refine and return
#                             #return self.step(path_segment,ti,(ti+tip1)/2.0,yi)
#                             t = (ti+t)/2.0
#                             distinct_solutions = False
#                             break

#             if approximate_solution and distinct_solutions:
#                 # finally, since we know that we have approximate solutions
#                 # that will converge to difference associated solutions we will
#                 # Netwon iterate
#                 yip1 = [ newton(df,x,yij) for yij in yi ]
#                 if t == tip1: break
#                 else: t = tip1

#         # return the t-point that we were able to step to as well as the
#         # y-roots lying above that t-point
#         return tip1, yip1


    def analytically_continue_segment(self, path_segment, t,
                                      Npts=4, checkpoint=None):
        """
        Analytically continue along a single segment of the path.

        A `RiemannSurfacePath` is a piecewise differentiable path
        composed of line segments and semi-circles. Each path segment
        is a function $\gamma : [0,1] \to \mathbb{C}$.

        Input:

        - 'path_segment_index': the index of the path segment

        - `t`: a value from 0 to 1.

        - `Npts`: (default: 4) number of interpolating points to use
          along the path. (The actual number of interpolating points
          used to analytically continue is based on the adaptive
          refinment performed using Smale's alpha theory.

        - `checkpoint`: (default: None) a tuple (ti,(xi,yi)) where
          (xi,yi) is the x-point and the y-fibre occuring at ti. If
          provided, analytic continuation will begin from this point.

        Output:

        A tuple (t,(x,y)) where x = x(t) and y is the analytically
        continued fibre.
        """
        # if integrating along a segment we require a checkpoint
        # (usually at t = 0) to indicate the proper ordering of the
        # fibre at this point along the global path
        path_segment_index = self._path_segments.index(path_segment)
        if checkpoint is None:
            t0,(xi,yi) = self._nearest_checkpoint(path_segment_index,t)
        else:
            t0,(xi,yi) = checkpoint

        # if the requested point is close to a checkpoint then do not
        # analytically continue
#         dt = numpy.double(t-t0)/Npts
#         ti = t0
#         while ti + dt < t:
#             tip1 = ti + dt
#             xi,yi = self.step(path_segment,ti,tip1,yi)
#         return xi,tuple(yi)
        tim1 = t0
        for ti in numpy.linspace(t0,t,Npts)[1:]:
            xi,yi = self.step(path_segment,tim1,ti,yi)
            tim1 = ti
        return xi,yi

    def analytically_continue(self, t, Npts=4):
        """
        """
        eps = 1e-15

        # The entire path is parameterized by t in [0,1]. We need to
        # determine which of the segments this t lies in.
        t_scaled = t * self._num_path_segments
        t_floor = numpy.floor(t_scaled)
        t_seg = t_scaled - t_floor

        # If the last point is requested (t=1) then decrement since it
        # should just be the last segment evaluated at t=1.
        t_floor = int(t_floor)
        if t_floor == self._num_path_segments:
            t_floor = -1
            t_seg = 1 - eps

        segment_index = t_floor
        return self.analytically_continue_segment(segment_index,Npts=Npts)


    def integrate(self,omega,x,y):
        """
        Integrates a differential omega = omega(x,y) along the path.
        """
        o = sympy.lambdify([x,y], omega, 'numpy')
        def integrand(t,seg=None):
            dxidt = seg[1](t)
            xi,yi = self.analytically_continue_segment(seg,t)
            return o(xi,yi[0]) * dxidt

        # For each segment, determine the proper ordering of the roots
        # at the start of the segment, define the integrand, and then
        # integrate along the segment
        re = numpy.double(0)
        im = numpy.double(0)
        for idx in range(self._num_path_segments):
            seg = self._path_segments[idx]
            re += scipy.integrate.romberg(lambda t: integrand(t,seg).real,0,1,
                                          tol=1e-14,rtol=1e-14,divmax=10)
            im += scipy.integrate.romberg(lambda t: integrand(t,seg).imag,0,1,
                                          tol=1e-14,rtol=1e-14,divmax=10)
        return re + 1.0j*im


    def plot_differential(self,omega,x,y,Npts=64):
        """
        Plot the differential along the path.
        """
        o = sympy.lambdify([x,y],omega,'numpy')
        ppseg = numpy.int(Npts/numpy.double(self._num_path_segments))
        tt = numpy.linspace(0,1,ppseg)

        def differential(t,seg=None):
            xi,yi = self.analytically_continue_segment(seg,t)
            return o(xi,yi[0])
        differential = numpy.vectorize(differential,excluded=['seg'])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        n = self._num_path_segments
        for k in range(n):
            seg = self._path_segments[k]
            oo = differential(tt,seg=seg)
            oo_re = oo.real
            oo_im = oo.imag
            ax.plot((tt+k)/n,oo_re,'r')
            ax.plot((tt+k)/n,oo_im,'b')

        ax.axis('tight')
        ax.set_xticks(numpy.arange(n+1,dtype=numpy.double)/n,minor=False)
        ax.grid(True)
        fig.show()


    def plot_integrand(self,omega,x,y,Npts=128):
        pass


    def sample_uniform(self, Npts=64, t0=0, t1=1, dxdt=False):
        """
        Return a uniform sample on the path.
        """
        if self.types == 'mpmath':
            t = sympy.mpmath.linspace(t0,t1,Npts)
        else:
            t = numpy.linspace(t0,t1,Npts).tolist()

        P = [self.analytically_continue(ti) for ti in t]
        return P


    def sample_clenshaw_curtis(self, Npts=64, t0=0, t1=1, dxdt=False):
        """
        Return a Clenshaw-Curtis sample (also referred to as a Chebysheb or
        cosine distribution sample) on the path.
        """
        if self.types == 'mpmath':
            pi = sympy.mpmath.pi
            theta = sympy.mpmath.linspace(0,pi,Npts)
            cos = sympy.mpmath.cos
        else:
            pi = numpy.pi
            theta = numpy.linspace(0,pi,Npts)
            cos = numpy.cos

        P = map(lambda phi: self.analytically_continue(cos(phi),dxdt=dxdt),
                theta)
        return P


    def decompose_points(self, P):
        """
        Converts a collection of Riemann surface points:..

            Ps = {..., (xi,yi), ...}

        to four lists

            { xi.real }, { xi.imag }, { yi.real }, { yi.imag }.
        """
        x,y = zip(*P)
        x_re = [xi.real for xi in x]
        x_im = [xi.imag for xi in x]

        y_re = []
        y_im = []
        for j in xrange(self.deg):
            y_re.append( [yi[j].real for yi in y] )
            y_im.append( [yi[j].imag for yi in y] )

        return x_re, x_im, y_re, y_im


    def plot_xpath(self, Npts=64, t0=0, t1=1, **kwds):
        """
        Plots the path in the complex x-plane.

        Inputs:

        - t0,t1: (default: 0,1) starting and ending points on the
          parameterized path.

        - Npts: (default: 64) number of interpolating points to plot.

        Additional keywords are sent to matplotlib.pyplot.plot().
        """
        tpts = numpy.linspace(t0,t1,Npts)
        xpts = numpy.array([self.get_x(ti) for ti in tpts],
                           dtype=numpy.complex)
        xre = xpts.real
        xim = xpts.imag

        fig = plt.figure()
        ax = fig.add_subplot(111)
        lc = color_plot_collection(xre, xim, cmap=cm.Blues, **kwds)
        ax.add_collection(lc)
        ax.axis('tight')

        fig.show()


    def plot(self, Npts=64, t0=0, t1=1, **kwds):
        """
        Plots the path in the complex x- and y-planes.

        Inputs:

        - t0,t1: (default: 0,1) Starting and ending point on the path.

        - Npts: (default: 64) Number of interpolating points to plot.

        - **kwds: additional keywords are sent to
            matplotlib.pyplot.plot()
        """
        deg = self.deg

        fig = plt.figure()

        P = self.sample_uniform(t0=t0,t1=t1,Npts=Npts)
        C = [Pi for ti,Pi in self._checkpoints]

        # plot x-points in the complex plane
        xax = fig.add_subplot(1,deg+1,1)
        xre, xim, yre, yim = self.decompose_points(P)
        lc = color_plot_collection(xre,xim,cmap=cm.Blues,**kwds)
        xax.add_collection(lc)
        xax.axis('tight')

        # plot each y-path in the complex plane
        yremin = numpy.Inf
        yremax = -numpy.Inf
        yimmin = numpy.Inf
        yimmax = -numpy.Inf
        for j in xrange(deg):
            yax = fig.add_subplot(1,deg+1,j+2)
            lc = color_plot_collection(yre[j],yim[j],cmap=cm.Greens,**kwds)
            yax.add_collection(lc)

            # adjust axes
            yremin = yremin if yremin < min(yre[j]) else min(yre[j])
            yimmin = yimmin if yimmin < min(yim[j]) else min(yim[j])
            yremax = yremax if yremax > max(yre[j]) else max(yre[j])
            yimmax = yimmax if yimmax > max(yim[j]) else max(yim[j])


        # plot checkpoints
        scale_factor = 1.1
        yremin *= scale_factor
        yimmin *= scale_factor
        yremax *= scale_factor
        yimmax *= scale_factor
        xre, xim, yre, yim = self.decompose_points(C)
        xax.plot(xre, xim, 'k.', **kwds)
        for j in xrange(deg):
            yax = fig.add_subplot(1,deg+1,j+2)
            yax.plot(yre[j], yim[j], 'k.', **kwds)
            yax.axis([yremin,yremax,yimmin,yimmax])

        fig.show()



    def plot3d(self, Npts=64, t0=0, t1=1, *args, **kwds):
        """
        Plots the path in the complex x- and y-planes.

        Inputs:

        - t0,t1: (default: 0,1) Starting and ending point on the path.

        - Npts: (default: 64) Number of interpolating points to plot.

        - **kwds: additional keywords are sent to
            matplotlib.axes3d.Axes3D.plot()

        """
        P = self.sample_uniform(t0=t0,t1=t1,Npts=Npts)
        x_re, x_im, y_re, y_im = self.decompose_points(P)
        zeros = numpy.zeros_like(x_re)
        tspace = numpy.linspace(0,1,len(x_re))

        fig = plt.figure(figsize=plt.figaspect(0.5))
        for i in xrange(self.deg):
            ax = fig.add_subplot(1, self.deg, i+1, projection='3d')
            ax.plot(y_re[i], y_im[i], tspace, 'g')
            ax.plot(y_re[i], y_im[i], zeros, color='grey')

            ax.plot((y_re[i][0], y_re[i][0]),
                    (y_im[i][0], y_im[i][0]),
                    (0,1), color='grey', linestyle='--')

        fig.show()



    def plot_path_segments(self, Npts=64, t0=0, t1=1, show_numbers=False,
                           *args,**kwds):
        """
        Plot
        """
        t_pts = sympy.mpmath.linspace(t0,t1,Npts)
        x_pts = map(self.get_x,t_pts)
        x_re = [x.real for x in x_pts]
        x_im = [x.imag for x in x_pts]

        fig = plt.figure()
        ax  = fig.add_subplot(1,1,1)

        eps = 0.1
        N = len(x_pts)
        ax.plot(x_re, x_im, 'b--', alpha=0.3)
        ax.hold(True)
        for n in xrange(N):
            ax.text(x_re[n], x_im[n], str(n), fontsize=8)

        ax.axis('tight')
        fig.show()




def plot_fibre(f,x,y,branch_point):
    """
    Plot
    """
    from monodromy import monodromy_graph
    G = monodromy_graph(f,x,y)
    base_point = G.node[0]['basepoint']
    base_sheets = G.node[0]['baselift']
    branch_points = [data['value'] for node,data in G.nodes(data=True)]
    for sheet in base_sheets:
        path_segments = path_around_branch_point(G,branch_point,1)
        gamma = RiemannSurfacePath((f,x,y),(base_point,sheet),
                                   path_segments=path_segments)
        gamma.plot(Npts=96)



if __name__=='__main__':
    print
    print "============================"
    print "=== Riemann surface path ==="
    print "============================"
    print

    import time
    import sympy
    import cProfile
    import pstats

    from sympy.abc import x,y

    # compute f and its derivatives with respect to y along with
    # additional data
    f1 = (x**2 - x + 1)*y**2 - 2*x**2*y + x**4
    f2 = -x**7 + 2*x**3*y + y**3
    f3 = (y**2-x**2)*(x-1)*(2*x-3) - 4*(x**2+y**2-2*x)**2
    f4 = y**2 + x**3 - x**2
    f5 = (x**2 + y**2)**3 + 3*x**2*y - y**3
    f6 = y**4 - y**2*x + x**2   # case with only one finite disc pt
    f7 = y**3 - (x**3 + y)**2 + 1
    f8 = (x**6)*y**3 + 2*x**3*y - 1
    f9 = 2*x**7*y + 2*x**7 + y**3 + 3*y**2 + 3*y
    f10= (x**3)*y**4 + 4*x**2*y**2 + 2*x**3*y - 1

    f = y**3 - x**3*y + x**7

    #
    # path #1 construction
    #
    z1 = numpy.complex(-2)
    z2 = numpy.complex(-1)
    z3 = numpy.complex(-1.5 + 1.0j)
    path_segments1 = [
        (lambda t,z1=z1,z2=z2: z1*(1-t) + z2*t,
         lambda t,z1=z1,z2=z2: z2-z1),
        (lambda t,z1=z2,z2=z3: z1*(1-t) + z2*t,
         lambda t,z1=z2,z2=z3: z2-z1),
        (lambda t,z1=z3,z2=z1: z1*(1-t) + z2*t,
         lambda t,z1=z3,z2=z1: z2-z1),
        ]
    x0 = z1
    y0 = polyroots(f,x,y,x0)

    print "=== (constructing path #1) ==="
    gamma1 = RiemannSurfacePath((f,x,y),(x0,y0),path_segments=path_segments1)

    print "=== (computing holomorphic differentials) ==="
    from abelfunctions.differentials import differentials
    omega = differentials(f,x,y)[0]
    print "omega =", omega

    print "=== (plotting differential on path #1) ==="
    gamma1.plot_differential(omega,x,y,Npts=64)


#     print "=== (integrating) ==="
#     cProfile.run("val = gamma1.integrate(omega,x,y)",'rs_path.profile')
#     p = pstats.Stats('rs_path.profile')
#     p.strip_dirs()
#     p.sort_stats('time').print_stats(12)
#     p.sort_stats('cumulative').print_stats(12)
#     p.sort_stats('calls').print_stats(12)
#     print "val =", val


#     #
#     # path #2 construction
#     #
#     z1 = numpy.complex(-2)
#     z2 = numpy.complex(-1)
#     z3 = numpy.complex(-1 - 0.5j)
#     z4 = numpy.complex(-2 - 0.5j)
#     path_segments2 = [
#         (lambda t,z1=z1,z2=z2: z1*(1-t) + z2*t,
#          lambda t,z1=z1,z2=z2: z2-z1),
#         (lambda t,z1=z2,z2=z3: z1*(1-t) + z2*t,
#          lambda t,z1=z2,z2=z3: z2-z1),
#         (lambda t,z1=z3,z2=z4: z1*(1-t) + z2*t,
#          lambda t,z1=z3,z2=z4: z2-z1),
#         (lambda t,z1=z4,z2=z1: z1*(1-t) + z2*t,
#          lambda t,z1=z4,z2=z1: z2-z1),
#         ]
#     x0 = z1
#     y0 = polyroots(f,x,y,x0)

#     print "=== (constructing path #2) ==="
#     gamma2 = RiemannSurfacePath((f,x,y),(x0,y0),path_segments=path_segments2)


#     print "=== (plotting) ==="
# #     P = gamma2.sample_uniform(32)
# #     xx,yy = zip(*P)
# #     xx = numpy.array(xx)
# #     y0,y1,y2 = map(numpy.array,zip(*yy))
# #     plt.plot(xx.real, xx.imag, 'k',
# #              y0.real, y0.imag, 'r.-',
# #              y1.real, y1.imag, 'g.-',
# #              y2.real, y2.imag, 'b.-')
# #     plt.show()


#     print "=== computing holom diffs ==="
#     from abelfunctions.differentials import differentials
#     omega = differentials(f,x,y)[0]

#     print "=== integrating along path #1 ==="
#     eps = 1e-15

#     print("integral:")
#     t = time.time()
#     print gamma1.integrate(omega,x,y)
#     print "time:", time.time()-t

#     print "\n=== integrating along path #2 ==="

#     print("integral:")
#     t = time.time()
#     print gamma2.integrate(omega,x,y)
#     print "time:", time.time()-t
