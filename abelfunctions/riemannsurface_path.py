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


import pdb


def factorial(n):
    return reduce(lambda a,b: a*b, xrange(1,n+1))

def smale_gamma(df,n,x,y):
    """
    Smale's gamma function.
    """
    mags = []
    for k in xrange(2,n+1):
        foo = (df[k](x,y)/df[1](x,y))/factorial(k)
        foo = numpy.abs(foo)**(1.0/(k-1.0))
        mags.append(foo)
    return max(mags)

def smale_beta(df,x,y):
    """
    Smale's beta function.
    """
    return numpy.abs(df[0](x,y)/df[1](x,y))

def smale_alpha(df,n,x,y):
    """
    Smale's alpha function. Used to determine if Newton's method will
    converge to a y-root.
    """
    return beta(df,x,y) * gamma(df,n,x,y)


def smale_newton(f,dfdy,xip1,y):
    eps = 1e-15
    maxsteps = 10
    for _ in xrange(maxsteps):
        step = f(xip1,y)/dfdy(xip1,y)
        if numpy.abs(step) < eps:
            break
        else:
            y -= step
    return y



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
            seg = lambda t,z0=z0,z1=z1: z0*(1-t) + z1*t
            dseg = lambda t,z0=z0,z1=z1: -z0+z1
        else:
            R,w,arg,d = map(cast_type,datum)
            seg = lambda t,R=R,w=w,arg=arg,d=d,exp=exp,j=j,pi=pi: \
                R*exp(j*(d*pi*t + arg)) + w
            dseg = lambda t,R=R,w=w,arg=arg,d=d,exp=exp,j=j,pi=pi: \
                (R*j*d*pi)*exp(j*(d*pi*t+arg))
        path_segments.append((seg,dseg))

    # Add the semicircle segments going around the branch point
    for datum in circle_data:
        R,w,arg,d = map(cast_type,datum)
        seg = lambda t,R=R,w=w,arg=arg,d=d,exp=exp,j=j,pi=pi: \
            R*exp(j*(d*pi*t + arg)) + w
        dseg = lambda t,R=R,w=w,arg=arg,d=d,exp=exp,j=j,pi=pi: \
            (R*j*d*pi)*exp(j*(d*pi*t+arg))
        path_segments.append((seg,dseg))

    # Add the reversed path segments leading back to the base point
    # (to reverse these paths, simply make the transformation t |-->
    # (1-t) )
    for datum in reversed(path_data):
        if len(datum) == 2:
            z0,z1 = map(cast_type,datum)
            seg = lambda t,z0=z0,z1=z1: z0*t + z1*(1-t)
            dseg = lambda t,z0=z0,z1=z1: z0-z1
        else:
            R,w,arg,d = map(cast_type,datum)
            seg = lambda t,R=R,w=w,arg=arg,d=d,exp=exp,j=j,pi=pi: \
                R*exp(j*(d*pi*(1-t) + arg)) + w
            dseg = lambda t,R=R,w=w,arg=arg,d=d,exp=exp,j=j,pi=pi: \
                -(R*j*d*pi) * exp(j*(d*pi*(1-t) + arg))
        path_segments.append((seg,dseg))

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

        self.P0 = (CC(P0[0]), map(CC,P0[1]))
        self.deg = sympy.degree(self.f,self.y)
        self.types = types

        # construct fast, type-cast lambda functions from the
        # algebraic curve. used in performing the Taylor step in the
        # analytic continuation
        dfdx = sympy.diff(self.f,self.x).expand()
        dfdy = sympy.diff(self.f,self.y).expand()
        self._f = sympy.lambdify((self.x,self.y), self.f, self.types)
        self.dfdx = sympy.lambdify((self.x,self.y), dfdx, self.types)
        self.dfdy = sympy.lambdify((self.x,self.y), dfdy, self.types)
        self.dfs = [sympy.lambdify((self.x,self.y),
                                   sympy.diff(self.f,self.y,k), self.types) 
                    for k in xrange(self.deg+1)]

        # path data
        self._path_segments     = path_segments
        self._num_path_segments = len(path_segments)

        # initialize the checkpoints: in each path segment
        # analytically continue to a certain number of points along
        # the path and store in memory. This is done so one doesn't
        # have to analytically continue from the base point every time
        # you want to find a point on the path
        self._checkpoints = [(0,self.P0)]
        self._initialize_checkpoints()


    def __repr__(self):
        return 'Path on the Riemann surface defined by the curve %s = 0'%self.f


    def _initialize_checkpoints(self):
        """
        Compute
        """
        ppseg = 4
        if self.types == 'mpmath':
            t_pts = sympy.mpmath.linspace(0,1,ppseg*self._num_path_segments)
        else:
            t_pts = numpy.linspace(0,1,ppseg*self._num_path_segments)

        tim1 = 0
        Pim1 = self.P0
        for ti in t_pts[1:]:
            Pi = self.analytically_continue(ti,Npts=8,checkpoint=(tim1,Pim1))
            self._checkpoints.append((ti,Pi))
            tim1 = ti
            Pim1 = Pi


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
        # on the path. By constuction, the checkpoints are ordered.
        for ti, Pi in self._checkpoints:
            if ti >= t:
                return (tim1, Pim1)
            else:
                tim1 = ti
                Pim1 = Pi

        # no suitable checkpoint found: either no checkpoints are
        # available or something wrong happened. Return the first
        # point of the path
        return self._checkpoints[0]


    def get_x(self, t, dxdt=False):
        """
        Returns the x-point corresponding to t.

        If the path is given in multiple path segments then some
        scaling is performed to determine which segment the
        corresponding x is computed from.
        """
        if self.types == 'mpmath':
            floor = sympy.mpmath.floor
            eps = sympy.mpmath.eps
        else:
            floor = numpy.floor
            eps = 1.0e-16

        # The entire path is parameterized by t in [0,1]. We need to
        # determine which of the segments this t lies in.
        t_scaled = t * self._num_path_segments
        t_floor = floor(t_scaled)
        t_seg = t_scaled - t_floor

        # If the last point is requested (t=1) then decrement since it
        # should just be the last segment evaluated at t=1.
        t_floor = int(t_floor)
        if t_floor == self._num_path_segments:
            t_floor = -1
            t_seg = 1 - eps

        seg, dseg = self._path_segments[t_floor]
        if dxdt:
# XXX
#            return self._num_path_segments * dseg(t_seg)
            return dseg(t_seg)
        else:
            return seg(t_seg)


    def __call__(self, t, dxdt=False):
        return self.analytically_continue(t,dxdt=dxdt)


    def analytically_continue(self, t, dxdt=False, Npts=8, checkpoint=None):
        """
        Analytically continue along the path to the given `t` in the
        interval [0,1]. self(0) returns the starting point.
        """
        n = self.deg
        alpha0 = (13.0 - 2.0*numpy.sqrt(17.0))/4.0
        

        # checkpointing allows us to start our analytic continuation
        # from a point further along the path from t=0
        if checkpoint:
            t0,(xi,yi) = checkpoint
        else:
            t0,(xi,yi) = self._nearest_checkpoint(t)

        # main loop
        maxdt = (t-t0)/Npts
        dt = maxdt
        ti = t0
        while ti < t:
            # iterate and approximate
            tip1 = ti + dt
            xip1 = self.get_x(tip1)
            yapp = [yij for yij in yi]

            # we have to check quadratic basin stuff for _every_ root
            # simultaneously or else we might have a branch jump
            are_approximate_solutions = True                
            beta_yapp = [smale_beta(self.dfs,xip1,yappj) for yappj in yapp]
            for j in xrange(n):
                # check if our guess is in the quadratic convergence basin
                gammaj = smale_gamma(self.dfs,n,xip1,yapp[j])
                alphaj = beta_yapp[j] * gammaj
                if alphaj >= alpha0:
                    are_approximate_solutions = False
                    break

            # only perform separated basins detection if the roots are
            # close enough
            separated_basins = True
            if are_approximate_solutions:
                # now that we know the solutions are approximate solutions, we
                # use the basin radius formula to ensure that each approximate
                # root is within only one quadradic convergence basin.
                for j in xrange(n):
                    betaj = beta_yapp[j]
                    for k in xrange(n):
                        if j != k:
                            dist = numpy.abs(yapp[j] - yapp[k])
                            betak = beta_yapp[k]

                            # use triangle inequality to guarantee separation
                            # since we don't know what the actual roots are
                            if dist <= 2*(betaj + betak):
                                separated_basins = False
                                break

            # raise warning if it looks like something went wrong
            if dt < 1e-15:
                raise Warning("Smallest adaptive refinement level " + \
                              "reached during analytic continuation.")

            if are_approximate_solutions and separated_basins:
                yi = [smale_newton(self.dfs[0],self.dfs[1],xip1,yappj) 
                      for yappj in yapp]
                ti = tip1
                xi = xip1
                dt = min(2*dt,maxdt)
            else:
                dt *= 0.5

        if dxdt:
            return (xi,tuple(yi)), self.get_x(ti,dxdt=True)
        else:
            return (xi,tuple(yi))



    def sample_uniform(self, t0=0, t1=1, Npts=64, dxdt=False):
        """
        Return a uniform sample on the path.
        """
        if self.types == 'mpmath':
            t = sympy.mpmath.linspace(t0,t1,Npts)
        else:
            t = numpy.linspace(t0,t1,Npts).tolist()

        P = [self.analytically_continue(ti,dxdt=dxdt) for ti in t]
        return P


    def sample_clenshaw_curtis(self, t0=0, t1=1, Npts=64, dxdt=False):
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
        x, y = zip(*P)
        x_re = [xi.real for xi in x]
        x_im = [xi.imag for xi in x]

        y_re = []
        y_im = []
        for j in xrange(self.deg):
            y_re.append( [yi[j].real for yi in y] )
            y_im.append( [yi[j].imag for yi in y] )

        return x_re, x_im, y_re, y_im


    def plot(self, t0=0, t1=1, Npts=64, **kwds):
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

        # plot chosen interpolating points
        x_ax = fig.add_subplot(1,deg+1,1)
        x_re, x_im, y_re, y_im = self.decompose_points(P)
        x_ax.plot(x_re, x_im, '-', **kwds)
        x_ax.plot(x_re[-1], x_im[-1], 'k.', markersize=30)
        x_ax.plot(x_re[0], x_im[0], 'r.', markersize=20)
        for j in xrange(deg):
            y_ax = fig.add_subplot(1,deg+1,j+2)
            y_ax.plot(y_re[j], y_im[j], 'g-', **kwds)
            y_ax.plot(y_re[j][-1], y_im[j][-1], 'k.', markersize=30)
            y_ax.plot(y_re[j][0], y_im[j][0], 'r.', markersize=20)

        # plot checkpoints
        x_re, x_im, y_re, y_im = self.decompose_points(C)
        x_ax.plot(x_re, x_im, 'k.', **kwds)
        for j in xrange(deg):
            y_ax = fig.add_subplot(1,deg+1,j+2)
            y_ax.plot(y_re[j], y_im[j], 'k.', **kwds)


        fig.show()



    def plot3d(self, t0=0, t1=1, Npts=128, *args, **kwds):
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




def plot_fibre(f,x,y,branch_point):
    """
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
    import sympy
    from sympy.abc import x,y
    from abelfunctions.monodromy import monodromy_graph, show_paths

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

    f12 = x**4 + y**4 - 1

    f = f12

    print "=== (computing monodromy graph) ==="
    bpt = 0
    G = monodromy_graph(f,x,y)
    path_segments = path_around_branch_point(G,bpt,1)
#    path_segments = path_around_infinity(G,1)

    print "===   (obtaining base place)    ==="
    base_point = G.node[0]['basepoint']
    base_sheets = G.node[0]['baselift']
    x0,y0 = base_point, base_sheets
    
    print "===     (constructing path)     ==="
    gamma = RiemannSurfacePath((f,x,y),(x0,y0),path_segments=path_segments)

    print "===        (plotting)           ==="
    gamma.plot(Npts=256)
