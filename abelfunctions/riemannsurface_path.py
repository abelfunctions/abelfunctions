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
import matplotlib.pyplot as plt



def polyroots(f,x,y,xi):
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
    coeffs = [c.evalf(subs={x:xi},n=dps) for c in p.all_coeffs()]
    coeffs = [sympy.mpmath.mpc(*(z.as_real_imag())) for z in coeffs]

    return sympy.mpmath.polyroots(coeffs)



def path_around_branch_point(G, bpt, rot):
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
            dir = -1 if prev_node in conjugates else 1 # XXX
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
        (next_radius, next_value, arg, dir),
        (next_radius, next_value, arg+sympy.mpmath.pi, dir)
        ]
    circle_data = circle_data * int(abs(rot))

    # 3) Use the path data to compute parameterizations of each of the
    # path segments.
    path_segments = []
    exp = sympy.mpmath.exp
    pi = sympy.mpmath.pi
    j = sympy.mpmath.j

    # Add the segments leading up to the branch point circle
    for datum in path_data:
        if len(datum) == 2:
            z0,z1 = map(sympy.mpmath.mpc,datum)
            seg = lambda t,z0=z0,z1=z1: z0*(1-t) + z1*t
            dseg = lambda t,z0=z0,z1=z1: -z0+z1
        else:
            R,w,arg,d = map(sympy.mpmath.mpc,datum)
            seg = lambda t,R=R,w=w,arg=arg,d=d: R*exp(j*(d*pi*t + arg)) + w
            dseg = lambda t,R=R,w=w,arg=arg,d=d: (R*j*d*pi)*exp(j*(d*pi*t+arg))
        path_segments.append((seg,dseg))

    # Add the semicircle segments going around the branch point
    for datum in circle_data:
        R,w,arg,d = map(sympy.mpmath.mpc,datum)
        seg = lambda t,R=R,w=w,arg=arg,d=d: R*exp(j*(d*pi*t + arg)) + w
        dseg = lambda t,R=R,w=w,arg=arg,d=d: (R*j*d*pi)*exp(j*(d*pi*t+arg))
        path_segments.append((seg,dseg))

    # Add the reversed path segments leading back to the base point
    # (to reverse these paths, simply make the transformation t |-->
    # (1-t) )
    for datum in reversed(path_data):
        if len(datum) == 2:
            z0,z1 = map(sympy.mpmath.mpc,datum)
            seg = lambda t,z0=z0,z1=z1: z0*t + z1*(1-t)
            dseg = lambda t,z0=z0,z1=z1: z0-z1
        else:
            R,w,arg,d = map(sympy.mpmath.mpc,datum)
            seg = lambda t,R=R,w=w,arg=arg,d=d: R*exp(j*(d*pi*(1-t) + arg)) + w
            dseg = lambda t,R=R,w=w,arg=arg,d=d: -(R*j*d*pi) * \
                exp(j*(d*pi*(1-t) + arg))
        path_segments.append((seg,dseg))
    
    return path_segments




class RiemannSurfacePath():
    """
    Defines a path on a Riemann surface parameterized on the interval
    [0,1]. Used for analytically continuing and integrating on Riemann
    surfaces.
    """
    def __init__(self, RS, P0, path_segments=None):
        """
        Create a path on the RiemannSurface, `RS`, starting at the
        RiemannSurfacePoint, `P0`.

        Input:

        - `RS`: a RiemannSurface on which the path is defined. (Or, a
          tuple containing `(f,x,y)` where ``f(x,y) = 0`` defined the
          Riemann surface.

        - `P0`: a RiemannSurfacePoint where the path begins.

        - `P1`: (optional) a RiemannSurfacePoint where the path ends.
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

        self.P0 = P0
        self.deg = sympy.degree(f,y)
        
        dfdx = sympy.diff(self.f,self.x).expand()
        dfdy = sympy.diff(self.f,self.y).expand()
        
        self._f = sympy.lambdify((self.x,self.y), self.f, "mpmath")
        self.dfdx = sympy.lambdify((self.x,self.y), dfdx, "mpmath")
        self.dfdy = sympy.lambdify((self.x,self.y), dfdy, "mpmath")

        self._checkpoint_cost   = 8
        self._path_segments     = path_segments
        self._num_path_segments = len(path_segments)

        self._checkpoints    = { 0:self.P0 }
        self._cache_size     = 1
        self._max_cache_size = 2**10

        self._initialize_checkpoints()


    def __repr__(self):
        return 'Path on the Riemann surface defined by the curve %s = 0'%self.f


    def _initialize_checkpoints(self):
        """
        Compute 
        """
        ppseg = 8
        t_pts = sympy.mpmath.linspace(0,1,ppseg*self._num_path_segments)
        for ti in t_pts:
            P = self.analytically_continue(ti,Npts=32)
            self._add_checkpoint(ti, P)


    def _nearest_checkpoint(self, t):
        """
        Returns the nearest checkpoint to the input, `t`.

        RiemannSurfacePath performs some caching of analytic
        continuation to save on the cost of numerical root finding.
        If `self.__call__()` requires many iterations to compute an
        analytic continuation then checkpoint data is saved in a cache
        making future analytic continuations potentially faster.
        """
        tim1 = 0
        Pim1 = self.P0

        # _checkpoints is a dictionary with keys ti and values points
        # on the path
        for ti, Pi in iter(sorted(self._checkpoints.items())):
            if ti >= t:
                return (tim1, Pim1)
            else:
                tim1 = ti
                Pim1 = Pi

        # no suitable checkpoint found: either no checkpoints are
        # available or something wrong happened. Return the first
        # point of the path
        return (0, self._checkpoints[0])



    def get_x(self, t, dxdt=False):
        """
        Returns the x-point corresponding to t.

        If the path is given in multiple path segments then some
        scaling is performed to determine which segment the
        corresponding x is computed from.
        """
        # The entire path is parameterized by t in [0,1]. We need to
        # determine which of the segments this t lies in.
        t_scaled = t * self._num_path_segments
        t_floor = sympy.mpmath.floor(t_scaled)
        t_seg = t_scaled - t_floor

        # If the last point is requested (t=1) then decrement since it
        # should just be the last segment evaluated at t=1.
        t_floor = int(t_floor)
        if t_floor == self._num_path_segments:
            t_floor = -1
            t_seg = 1-sympy.mpmath.eps

        seg, dseg = self._path_segments[t_floor]
        if dxdt:    
            return self._num_path_segments * dseg(t_seg)
        else:
            return seg(t_seg)

        

    def _add_checkpoint(self, ti, Pi):
        """
        Adds the checkpoint Pi = {xi = x(ti), yi = y(x(ti))} to the
        inner cache. If the max cache size is reached then the first
        key tj found that is less than ti is removed from the
        checkpoint cache.
        """
        self._checkpoints[ti] = Pi
        self._cache_size += 1

        # pop an item from the cache
        if self._cache_size > self._max_cache_size:
            for tj in self._checkpoints.iterkeys():
                if tj < ti:
                    self._checkpoints.pop(tj)
                    self._cache_size -= 1
                    break

        
    def _clear_checkpoints(self):
        """
        Empties the checkpoint cache.
        """
        self._checkpoints.clear()
        self._checkpoints = { 0:self.P0 }
        self._cache_size = 1

        

    def __call__(self, t, dxdt=False):
        return self.analytically_continue(t,dxdt=dxdt)



    def analytically_continue(self, t, dxdt=False, Npts=16):
        """
        Analytically continue along the path to the given `t` in the
        interval [0,1]. self(0) returns the starting point. 
        """
        eps = sympy.mpmath.eps
        deg = self.deg

        # get the nearest already computed point on the path. If the
        # nearest computed point is at "t" then just return the cached
        # point now.
        t0, P0 = self._nearest_checkpoint(t)
        if t == t0:
            if dxdt:
                return P0, self.get_x(t,dxdt=True)
            else:
                return P0

        # Analytic continuation loop: start with dt = 1. Take a step
        # using Taylor series. Use a root finder to get to the closest
        # point on the Riemann surface. Check if this point is on the
        # correct sheet. If not, have the dt and try again
        tim1 = t0
        xim1 = P0[0]
        yim1 = P0[1]

        maxiter = 32
        maxn = 0
        t_pts = sympy.mpmath.linspace(t0,t,Npts)
        for i in xrange(1,Npts):
            ti = t_pts[i]
            xi = self.get_x(ti)

            # Taylor step to obtain approximate
            dx = xi-xim1
            dy = - dx * self.dfdx(xim1,yim1) / self.dfdy(xim1,yim1)
            yi_approx = yim1 + dy

            # Newton iterate to next point. (Note: this is done
            # instead of "sympy.mpmath.polyroots" for speed and
            # instead of sympy.mpmath.findroot for performance.)
            #
            # Add extra precision, too
            yi = yi_approx
            with sympy.mpmath.extraprec(4):
                for n in xrange(maxiter):
                    step = self._f(xi,yi) / self.dfdy(xi,yi)
                    if sympy.mpmath.absmin(step) < eps:
                        maxn = max(n, maxn)
                        break
                    yi -= step

            xim1 = xi
            yim1 = yi


        # Checkpoint this point if it took too many Newton iterations
        # at any point along the path.
        if maxn >= 6:
            self._add_checkpoint(t,(xi,yi))

        if dxdt:
            return (xi,yi), self.get_x(t,dxdt=True)
        else:
            return (xi,yi)



    def sample_uniform(self, t0=0, t1=1, Npts=64, dxdt=False):
        """
        Return a uniform sample on the path.
        """
        t = sympy.mpmath.linspace(t0,t1,Npts)
        P = map(lambda ti: self.analytically_continue(ti,dxdt=dxdt), t)
        return P


    def sample_clenshaw_curtis(self, t0=0, t1=1, Npts=64, dxdt=False):
        """
        Return a Clenshaw-Curtis sample (also referred to as a Chebysheb or
        cosine distribution sample) on the path.
        """
        pi = sympy.mpmath.pi
        theta = sympy.mpmath.linspace(0,pi,Npts)
        P = map(lambda phi: self.analytically_continue(sympy.mpmath.cos(phi),
                                                       dxdt=dxdt),
                theta)
        return P


    def decompose_points(self, P):
        """
        Converts a collection of Riemann surface points:..

            Ps = {..., (xi,yi), ...}

        to four lists
        
            { xi.real }, { xi.imag }, { yi.real }, { yi.imag }.
        """
        x, y = zip(*P)
        x_re = [xi.real for xi in x]
        x_im = [xi.imag for xi in x]
        y_re = [yi.real for yi in y]
        y_im = [yi.imag for yi in y]
        
        return x_re, x_im, y_re, y_im
        

    def plot(self, t0=0, t1=1, Npts=64, show_numbers=False, **kwds):
        """
        Plots the path in the complex x- and y-planes.

        Inputs:

        - t0,t1: (default: 0,1) Starting and ending point on the path.

        - Npts: (default: 64) Number of interpolating points to plot.
        
        - show_numbers: (default: False) If true, will plot the index
        of the points on the path as well. This helps when trying to
        determine the direction of the path.

        - **kwds: additional keywords are sent to
            matplotlib.pyplot.plot()
        """
        P = self.sample_uniform(t0=t0,t1=t1,Npts=Npts)
        
        fig = plt.figure()
        x_ax = fig.add_subplot(2,1,1)
        y_ax = fig.add_subplot(2,1,2)

        # First, plot all checkpoints.
        checkpoints = self._checkpoints.values()
        x_re, x_im, y_re, y_im = self.decompose_points(checkpoints)

        x_ax.plot(x_re, x_im, '.', **kwds)
        y_ax.plot(y_re, y_im, '.', **kwds)
        
        
        # Second, plot requested interpolants
        x_re, x_im, y_re, y_im = self.decompose_points(P)
        x_ax.plot(x_re[0], x_im[0], 'k.', markersize=20, **kwds)
        y_ax.plot(y_re[0], y_im[0], 'k.', markersize=20, **kwds)
        x_ax.plot(x_re, x_im, **kwds)
        y_ax.plot(y_re, y_im, **kwds)
        if show_numbers:
            for n in xrange(len(y_re)):
                if not y_only:
                    x_ax.text(x_re[n], x_im[n], str(n), fontsize=10)
                y_ax.text(y_re[n], y_im[n], str(n), fontsize=10)

        x_ax.axis('tight')
        y_ax.axis('tight')

        fig.show()



    def plot3d(self, t0=0, t1=1, Npts=64, **kwds):
        """
        Plots the path in the complex x- and y-planes.

        Inputs:

        - t0,t1: (default: 0,1) Starting and ending point on the path.

        - Npts: (default: 64) Number of interpolating points to plot.

        - **kwds: additional keywords are sent to
            matplotlib.axes3d.Axes3D.plot()

        """
        t_pts = sympy.mpmath.linspace(t0,t1,Npts)
        P_pts = self.sample_uniform(t0=t0,t1=t1,Npts=Npts)
        x_pts, y_pts = zip(*P_pts)

        x_re = [x.real for x in x_pts]
        x_im = [x.imag for x in x_pts]
        y_re = [y.real for y in y_pts]
        y_im = [y.imag for y in y_pts]

        fig = plt.figure(figsize=plt.figaspect(0.5))

        # Axis #1: real part of path
        y_re_min = min(y_re)
        z = [y_re_min]*Npts
        ax1 = fig.add_subplot(2,1,1,projection='3d')
        ax1.plot(x_re,x_im,z,*args,**kwds)
        ax1.plot(x_re,x_im,y_re,*args,**kwds)

        # Axis #1: real part of path
        y_im_min = min(y_im)
        z = [y_im_min]*Npts
        ax2 = fig.add_subplot(2,1,2,projection='3d')
        ax2.plot(x_re,x_im,z,*args,**kwds)
        ax2.plot(x_re,x_im,y_im,*args,**kwds)

        fig.show()



    def plot_path_segments(self, t0=0, t1=1, Npts=64, show_numbers=False,
                           *args,**kwds):
        """
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
        
        
        

