"""
Riemann Surfaces
"""

import numpy
import scipy
import sympy
import networkx as nx

from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt


#from abelfunctions.differentials import differentials
from monodromy   import Monodromy
from homology    import homology


import pdb



def differentials(foo):
    return None



def polyroots(f,x,y,xi):
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

        - `RS`: a RiemannSurface on which the path is defined

        - `P0`: a RiemannSurfacePoint where the path begins.

        - `P1`: (optional) a RiemannSurfacePoint where the path ends.
        """
        self.RS = RS
        self.P0 = P0

        self.f = RS.f
        self.x = RS.x
        self.y = RS.y
        
        dfdx = sympy.diff(f,x).expand()
        dfdy = sympy.diff(f,y).expand()
        
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
        return 'Path on %s' %(self.RS)


    def _initialize_checkpoints(self):
        """
        Compute 
        """
        ppseg = 8
        t_pts = sympy.mpmath.linspace(0,1,ppseg*self._num_path_segments)
        for ti in t_pts:
            P = self.analytically_continue(ti,Npts=16)
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
            return dseg(t_seg)
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



    def analytically_continue(self, t, dxdt=False, Npts=4):
        """
        Analytically continue along the path to the given `t` in the
        interval [0,1]. self(0) returns the starting point. 
        """
        eps = sympy.mpmath.eps
        deg = self.RS.deg

        # get the nearest already computed point on the path. If the
        # nearest computed point is at "t" then just return the cached
        # point now.
        t0, P0 = self._nearest_checkpoint(t)
        if t == t0:
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
        if maxn >= 4:
            self._add_checkpoint(t,(xi,yi))

        if dxdt:
            return (xi,yi), self.get_x(t,dxdt=True)
        else:
            return (xi,yi)



    def sample_uniform(self, t0=0, t1=1, Npts=64):
        """
        Return a uniform sample on the path.
        """
        t = sympy.mpmath.linspace(t0,t1,Npts)
        P = map(self.analytically_continue, t)
        return P


    def sample_clenshaw_curtis(self, t0=0, t1=1, Npts=64):
        """
        Return a Clenshaw-Curtis sample (also referred to as a Chebysheb or
        cosine distribution sample) on the path.
        """
        pi = sympy.mpmath.pi
        theta = sympy.mpmath.linspace(0,pi,Npts)
        P = map(lambda phi: self.analytically_continue(sympy.mpmath.cos(phi)),
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
        x_ax.plot(x_re, x_im, 'o', **kwds)
        y_ax.plot(y_re, y_im, 'o', **kwds)
        
        
        # Second, plot requested interpolants
        x_re, x_im, y_re, y_im = self.decompose_points(P)

        x_ax.plot(x_re, x_im, **kwds)
        y_ax.plot(y_re, y_im, **kwds)
        if show_numbers:
            for n in xrange(len(y_re)):
                x_ax.text(x_re[n], x_im[n], str(n), fontsize=8)
                y_ax.text(y_re[n], y_im[n], str(n), fontsize=8)

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
        
        
        

        
        
    
        

class RiemannSurface(Monodromy):
    """
    Class for defining a Riemann surface corresponding to a plane
    algebraic curve.
    """
    def __init__(self, f, x, y):
        # XXX clean this up. Understand subclassing better.
        super(RiemannSurface, self).__init__(f,x,y)
        self._monodromy = self.monodromy()
        self._homology  = homology(f,x,y)


    def __repr__(self):
        return "Riemann surface defined by the algebraic curve %s." %(self.f)



    def __call__(self, x, y):
        return RiemannSurfacePoint(self, x, y)



    def point(self, x0, y0):
        """
        Returns the point on the Riemann surface closest to
        `(x0,y0)`. (In the event that floating point precision is used
        to calculate the coordinates `(x0,y0)`.
        """
        return RiemannSurface_Point(self,x0,y0)
    


    def holomorphic_differentials(self):
        """ 
        Returns the basis of holomorphic differentials defined on the
        Riemann surface.
        """
        return differentials(f,x,y)



    def a_cycle(self, i):
        """
        Returns the ith a_cycle as an array of interpolating points
        starting from and ending at the basepoint of the Riemann
        surface.
        """
        pass


    
    def b_cycle(self, i):
        """
        Returns the ith b_cycle as an array of interpolating points
        starting from and ending at the basepoint of the Riemann
        surface.
        """
        pass



    def c_cycle(self, i):
        """
        Returns a path in the complex x-plane of the x-points in the
        monodromy path.
        
        Input:

        - i: the index of the c-cycle
        
        Output:

        - a RiemannSurfacePath parameterizing the c-cycle
        """
        # get the cycle data from homology: each c-cycle is a list of
        # alternating sheet numbers s_k and branch point / number of
        # rotations tuples (b_{i_k}, n_k)
        cycles  = self._homology['cycles']
        
        G = self.monodromy_graph()
        root = G.node[0]['root']
        path_segments = []

        # For each (branch point, rotation number) pair appearing in
        # the cycle compute the path segments the complex x-plane
        # going around the given branch points a number of times equal
        # to the rotation number. Add these segments to the list of
        # path segemnts.
        for (bpt, rot) in cycles[i][1::2]:
            bpt_path_segments = path_around_branch_point(G, bpt, rot)
            path_segments.extend(bpt_path_segments)

        # Construct the RiemannSurfacePath
        x0 = self.base_point()
        y0 = self.base_lift()[0]  # c-cycles always start on sheet 0
        gamma = RiemannSurfacePath(self, (x0,y0), path_segments=path_segments)

        return gamma



    def integrate(self, omega, x, y, path):
        """
        Integrates the differential `omega`, defined on the
        Riemann surface, on the RiemannSurfacePath `path`.
        
        Input:

        - `omega,x,y`: a Sympy funciton in the variables `x` and `y`

        - `path`: a RiemannSurfacePath defined on the Riemann surface
        """
        x0,y0 = path(0)
        omega = sympy.lambdify((x,y), omega, "mpmath")
        
        # Numerically integrate over each path segment. This is most
        # probably a good idea since it allows for good checkpointing.
        def integrand(t):
            (xi,yi),dxdt = path(t, dxdt=True)
            return omega(xi,yi) * dxdt

        val = sympy.mpmath.mpc(0)
        n   = sympy.mpmath.mpf(path._num_path_segments)
        for k in xrange(n):
            val += sympy.mpmath.quadgl(integrand, [k/n,(k+1)/n])
        
        return val



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
        lincombs = self._homology['linearcombination']
        n_cycles = len(self._homology['cycles'])
        c_cycles = [self.c_cycle(i) for i in range(n_cycles)]
        differentials = self.differentials()
        x = self.x
        y = self.y

        # XXX temporary...still need to figure out good way to do
        # this...
        A = []
        B = []
        for i in xrange(self.genus()):
            omega = differentials[i]
            Ai = []
            Bi = []
            for j in xrange(g):
                lincomb  = lincombs[j]
                # XXX check that len(lincomb) == n_cycles
                integral = sum(lincomb[k]*self.integrate(omega,x,y,c_cycles[k]) 
                               for k in xrange(n_cycles) if lincomb[k] != 0)
                Ai.append(integral)

                lincomb  = lincombs[j+g]
                integral = sum(lincomb[k]*self.integrate(omega,x,y,c_cycles[k]) 
                               for k in xrange(n_cycles) if lincomb[k] != 0)
                Bi.append(integral)

            A.append(Ai)
            B.append(Bi)

        # need to transpose...
        return (A,B)



if __name__ == '__main__':
    from sympy.abc import x,y

    f0 = y**3 - 2*x**3*y - x**8  # Klein curve

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

    f11= y**2 - x*(x-1)*(x-2)*(x-3)  # simple genus one hyperelliptic


    f = f11
    X = RiemannSurface(f,x,y)


