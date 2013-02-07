"""
Riemann Surfaces
"""

import numpy
import scipy
import sympy
import networkx as nx
import matplotlib.pyplot as plt

#from abelfunctions.differentials import differentials
from monodromy   import Monodromy
from homology    import homology


def differentials(foo):
    return None


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
            dir = 1 if curr_node in conjugates else -1
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

    def __getitem__(self,i):
        if i: return self.y
        else: return self.x



class RiemannSurfacePath():
    """
    Defines a path on a Riemann surface parameterized on the interval
    [0,1]. Used for analytically continuing and integrating on Riemann
    surfaces.
    """
    def __init__(self, RS, P0, P1=None, path_segments=None):
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
        self.P1 = P1

        self._checkpoint_cost   = 3
        self._path_segments     = path_segments
        self._num_path_segments = len(path_segments)

        self._checkpoints    = { 0:(self.P0.x, self.P0.y) }
        self._cache_size     = 1
        self._max_cache_size = 2**10


    def __repr__(self):
        if self.P1:
            return 'Path from %s to %s on %s'%(self.P0,self.P1,self.RS)
        else:
            return 'Closed path at %s on %s'%(self.P0,self.RS)


    def _from_path_segments(self, path_segments):
        """
        Construct the path from a set of "path_segments". These are used
        internally to define a piecewise differentiable path in the 
        complex x-plane.
        """
        self._path_segments = path_segments
        self._num_path_segments = len(path_segments)

        self.P1 = self(1)


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
        xim1 = self.P0.x
        yim1 = self.P0.y

        # _checkpoints is a dictionary with keys ti and values points
        # on the path
        for ti, (xi, yi) in self._checkpoints.iteritems():
            if ti >= t:
                return (tim1, (xim1, yim1))
            else:
                tim1 = ti
                xim1 = xi
                yim1 = yi

        # no suitable checkpoint found: either no checkpoints are
        # available or something wrong happened. Return the first
        # point of the path
        return (0, self._checkpoints[0])


    def get_x(self, t):
        """
        Returns the x-point corresponding to t.

        If the path is given in multiple path segments then some
        scaling is performed to determine which segment the
        corresponding x is computed from.
        """
        t_scaled = t*self._num_path_segments
        t_floor = sympy.mpmath.floor(t_scaled)
        t = t_scaled - t_floor

        # If the last point is requested (t=1) then decrement since it
        # should just be the last segment evaluated at t=1.
        t_floor = int(t_floor)
        if t_floor == self._num_path_segments:
            t_floor -= 1

        return self._path_segments[t_floor][0](t)

        

    def _add_checkpoint(self, ti, xi, yi):
        """
        Adds the checkpoint xi = x(ti), yi = y(x(ti)) to the inner
        cache. If the max cache size is reached then the first key tj
        found that is less than ti is removed from the checkpoint
        cache.
        """
        self._checkpoints[ti] = (xi,yi)

        # pop an item from the cache7
        if self._cache_size == self._max_cache_size:
            for tj in self._checkpoints.iterkeys():
                if tj < ti:
                    self._checkpoints.pop(ti)
                    break

        

    def _clear_checkpoints(self):
        """
        Empties the checkpoint cache.
        """
        self._checkpoints.clear()
        self._checkpoints = { 0:(self.P0.x, self.P0.y) }
        self._cache_size = 1
        


    def __call__(self, t, dxdt=False):
        """
        Analytically continue along the path to the given `t` in the
        interval [0,1]. self(0) returns the starting point. 
        """
        eps = sympy.mpmath.eps

        # get the nearest already computed point on the path. If the
        # nearest computed point is at "t" then just return the cached
        # point now.
        t0, (x0,y0) = self._nearest_checkpoint(t)
        if t == t0:
            return (x0,y0)

        # Analytic continuation loop: start with dt = 1. Take a step
        # using Taylor series. Use a root finder to get to the closest
        # point on the Riemann surface. Check if this point is on the
        # correct sheet. If not, have the dt and try again
        tim1 = t0
        ti   = t
        xim1 = x0
        yim1 = y0
        xi   = self.get_x(ti)

        # keep track of how many times we need to refine the grid for
        # checkpointing purposes
        num_refinements = 0   
#        while tim1 != t:
        while sympy.mpmath.absmin(tim1-t) > eps:
            dx   = xi-xim1
            f_xi  = lambda y: self.RS.f(xi,y)
            df_xi = lambda y: - dx * self.RS.dfdx(xi,y) / self.RS.dfdy(xi,y)

            # Take a continuation step using a first order Taylor
            # series.  (The root finder needs a 3-tuple guess so we
            # add a little bit of error to dy.)
            yp        = - self.RS.dfdx(xim1,yim1) / self.RS.dfdy(xim1,yim1)
            dy        = yp * dx
            yi_approx = yim1 + dy
            guess     = (yi_approx - dy - eps, yi_approx, yi_approx + dy + eps)
            yi        = sympy.mpmath.findroot(f_xi, guess, df=df_xi,
                                              solver='muller')

            # Check if we're on the correct branch: yi (the point on
            # the RS) should be closer to yi_approx (the Taylor approx
            # of yi) than to yim1 (the previous point). If we're
            # further away then the root finder had trouble.
            #
            # XXX Not sure if this is the right thing to do...
            if sympy.mpmath.absmin(yi-yi_approx)<sympy.mpmath.absmax(yi-yim1):
                # we're close enough
                tim1 = ti
                ti   = t
                xim1 = xi
                yim1 = yi
                xi   = self.get_x(ti)
            else:
                # refine
                ti = (ti + tim1) / 2.0
                xi = self.get_x(ti)
                num_refinements += 1

        # add checkpoint if we needed to refine too many times
        if num_refinements >= self._checkpoint_cost:
            self._add_checkpoint(ti,xi,yi)

        Pi = RiemannSurfacePoint(self.RS,xi,yi,exact=True)
        if dxdt: 
            return Pi, dx/dt
        else:
            return Pi


    def sample_uniform(self, Npts):
        """
        Return a uniform sample on the path.
        """
        t_pts = sympy.mpmath.linspace(0,1,Npts)
        P_pts = map(self, t_pts)
        return P_pts


    def plot(self, *args, **kwds):
        """
        Plots the path in the complex x- and y-planes.
        """
        # get the number of points to plot
        try: 
            Npts = kwds['Npts']
        except KeyError:
            Npts = 32

        P_pts = self.sample_uniform(Npts)
        x_pts, y_pts = zip(*P_pts)
        x_re = [x.real for x in x_pts]
        x_im = [x.imag for x in x_pts]
        y_re = [y.real for y in y_pts]
        y_im = [y.imag for y in y_pts]

        fig = plt.figure()
        x_ax = fig.add_subplot(1,2,1)
        y_ax = fig.add_subplot(1,2,2)

        x_ax.plot(x_re, x_im, *args, **kwds)
        y_ax.plot(y_re, y_im, *args, **kwds)

        x_ax.axis('tight')
        y_ax.axis('tight')

        fig.show()

    def plot_path_segments(self, *args, **kwds):
        # get the number of points to plot
        try: 
            Npts = kwds['Npts']
        except KeyError:
            Npts = 64

        t_pts = sympy.mpmath.linspace(0,1,Npts)
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
            if n < N/2: eps = -0.01
            else:       eps = 0.01

            ax.text(x_re[n], x_im[n]+eps, str(n), fontsize=8)

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

        _dfdx = sympy.diff(f, x).simplify()
        _dfdy = sympy.diff(f, y).simplify()

        # Fast, multipreicise evaluation functions
        self.p    = sympy.Poly(f,[x,y])
        self.f    = sympy.lambdify([x,y], f, "mpmath")
        self.dfdx = sympy.lambdify([x, y], _dfdx, "mpmath")
        self.dfdy = sympy.lambdify([x, y], _dfdy, "mpmath")
        self.x = x
        self.y = y
        
        self.n = sympy.degree(f,y)



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
        P = RiemannSurfacePoint(self, x0, y0, exact=True)
        gamma = RiemannSurfacePath(self, P, path_segments=bpt_path_segments)

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
        omega = sympy.lambdify([x,y], omega)
        
        # Numerically integrate over each path segment. This is most
        # probably a good idea since it allows for good checkpointing.
        def func(t):
            P,dxdt = path(t, dxdt=True)
            return omega(P.x, P.y) * dxdt

        val = sympy.mpmath.mpc(0)
        n   = sympy.mpmath.mpf(path._num_path_segments)
        for k in xrange(n):
            val += quad(func, [k/n,(k+1)/n])
            
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


