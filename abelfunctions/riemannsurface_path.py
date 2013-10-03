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

from itertools import tee

import pdb

def factorial(n):
    return reduce(lambda a,b: a*b, xrange(1,n+1))

def newton(df,xip1,yij):
    step = 1
    while numpy.abs(step) > 1e-14:
        # check if Df is invertible. (If not, then we are at a
        # critical point.)
        df1 = df[1](xip1,yij)
        if numpy.abs(df1) < 1e-14:
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

smale_alpha0 = numpy.double(13.0 - 2.0*numpy.sqrt(17.0))/4.0
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


def _path_segments_from_path_data(path_data,circle_data,types='numpy'):
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
    Returns a list of tuples encoding information about the x-part of the
    Riemann Surfaace path starting from the base point going around the branch
    point, "bpt", "rot" number of times, and then returning to the starting
    point.

    Line segments are encoded as tuples (z0,z1) where z0 is the starting
    x-value and z1 is the ending x-value.

    Semicircles are encoded as tuples (R,w,arg,dir) where R is the radius, w is
    the center, arg is the starting argument (e.g. arg=0 means start on the
    right side of the circle), and dir indicates which direction to travel
    around the circle in a semicircular arc.

    Input:

    - G: the "monodromy graph", as computed by monodromy_graph()

    - bpt: the index of the target branch point

    - rot: the rotation number and direction of the path going around
    branch point "bpt".

    Output:

    A list of tuples encoding the path information.
    """
    abs = numpy.abs
    pi = numpy.pi

    # retreive the vertices between the base point vertex and the target
    # vertex.
    root = G.node[0]['root']
    path_vertices = nx.shortest_path(G, source=root, target=bpt)

    # retreive the conjugate nodes. a conjugate node is one that we traverse
    # clockwise (negative direction) instead of counter-clockwise (positive
    # direction)
    conjugates = G.node[bpt]['conjugates']

    # (1) Compute path data for semi-circle / line pairs leading to the circle
    # encircling the target branch point. (Taking conjugates into account.)
    # Conjugation will indicate whether to pass a vertex on the path towards
    # the target either above or below the vertex.
    path_data = []
    prev_node = root
    for idx in range(len(path_vertices)-1):
        curr_node   = path_vertices[idx]
        curr_radius = G.node[curr_node]['radius']
        curr_value  = G.node[curr_node]['value']

        next_node   = path_vertices[idx+1]
        next_radius = G.node[next_node]['radius']
        next_value  = G.node[next_node]['value']

        # Determine if semi-circle is needed. This is done by checking the path
        # index of the previous edge with the path index of the next edge. If
        # needed, add semicircle going in the appropriate direction where the
        # direction is determined by the conjugation list. A special case is
        # taken if we're at the root vertex.
        curr_edge_index = G[curr_node][next_node]['index']
        if prev_node == root:
            prev_edge_index = (0,-1)  # in the root node case
        else:
            prev_edge_index = G[prev_node][curr_node]['index']
        if prev_edge_index[1] != curr_edge_index[0]:
            arg = pi if prev_edge_index[1] == -1 else 0
            dir = -1 if curr_node in conjugates else 1 # XXX
            path_data.append((curr_radius, curr_value, arg, dir))

        # Add the line to the next discriminant point.
        start = curr_value + curr_edge_index[0]*curr_radius
        end   = next_value + curr_edge_index[1]*next_radius
        path_data.append((start,end))

        # Update previous point
        prev_node = curr_node

    # (2) Construct interpolating points around the target branch point. The
    # rotation number "rot" tells us how many times to go around the branch
    # point and in which direction. There's a special case for when we just
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

    # (3) Construct the reverse path data. This is just the path defined in
    # part (1) above but traversed in teh reverse direction. This is just an
    # appropriate transformation on the path segment data.
    reversed_path_data = []
    for datum in reversed(path_data):
        if len(datum) == 2:
            z0,z1 = datum
            reversed_path_data.append( (z1,z0) )
        else:
            R,w,arg,dir = datum
            reversed_path_data.append( (R,w,arg+pi,-dir) )

    return path_data + circle_data + reversed_path_data


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
    abs = numpy.abs
    arg = numpy.angle
    pi = numpy.pi
    CC = numpy.complex

    # determine the center of the circle encircling the entire graph
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
    if abs(base_point - z1) > 1e-14:
        path_data = [(base_point,z1)]
    else:
        path_data = []

    circle_data = [(radius,center,arg0,dir), (radius,center,arg0+pi,dir)]
    circle_data = circle_data * int(abs(rot))

    # (3) Construct the reverse path data. This is just the path defined in
    # part (1) above but traversed in teh reverse direction. This is just an
    # appropriate transformation on the path segment data.
    if path_data == []:
        reversed_path_data = []
    else:
        reversed_path_data = [(z1,base_point)]

    return path_data + circle_data + reversed_path_data




class RiemannSurfacePathSegment():
    """
    Defines a segment of a Riemann surface path parameterized on the interval
    [0,1]. Used for analytically continuing and integrating on Riemann
    surfaces.

    All Riemann surface paths have either line segments or semi-circles as the
    x-plane part of the path.
    """
    def __init__(self,RS,P0,n_checkpoints=10):
        """
        Construct a RiemannSurfacePathSegment

        Input:

        - RS: a RiemannSurface

        - P0: a tuple consisting of the x-value and ordered y-fibre at the
          start of the path

        - n_checkpoints: (default: 8) number of checkpoints along the path so
          analytic continuation doesn't have to start from the beginning of the
          path
        """
        # HACK. clean this up later. monodromy is the only function that calls
        # RiemannSurfacePath wtihout first constructing a RiemannSurface
        # object. Consider making monodromy inaccessible outside of
        # RiemannSurface. (mondromy and homology are internal, anyway...)
        if isinstance(RS,tuple):
            self.RS = RS
            self.f,self.x,self.y = RS
        else:
            self.RS = RS
            self.f = RS.f
            self.x = RS.x
            self.y = RS.y

        self.x0 = numpy.complex(P0[0])
        self.y0 = tuple(map(numpy.complex,P0[1]))
        self.P0 = (self.x0,self.y0)

        self.deg = sympy.degree(self.f,self.y)
        self.df = [
            sympy.lambdify((self.x,self.y),sympy.diff(self.f,self.y,k),'numpy')
            for k in range(self.deg+1)
            ]

        self._checkpoints = [(numpy.double(0.0),self.P0)]
        self._n_checkpoints = n_checkpoints
        self._initialize_checkpoints()


    def __str__(self):
        return r'RiemannSurfacePathSegment on the curve ' + \
            'f(%s,%s) = %s'%(self.x,self.y,self.f)

    def __call__(self,t,checkpoint=None):
        return self.analytically_continue(t,checkpoint=checkpoint)

    def _initialize_checkpoints(self):
        """
        Analytically continue along the entire path and record checkpoint
        places along the way.
        """
        t_pts = numpy.linspace(0,1,self._n_checkpoints,endpoint=True)
        tim1,Pim1 = self._checkpoints[0]
        for ti in t_pts[1:]:
            Pi = self.analytically_continue(ti,checkpoint=(tim1,Pim1))
            self._checkpoints.append((ti,Pi))
            tim1 = ti
            Pim1 = Pi

    def _nearest_checkpoint(self,t):
        """
        Returns the checkpoint closest to, but occuring before, t.
        """
        tim1 = numpy.double(0.0)
        Pim1 = self.P0
        for ti,Pi in self._checkpoints:
            if ti >= t:
                return tim1,Pim1
            else:
                tim1 = ti
                Pim1 = Pi

        # use the base point if a valid checkpoint isn't found.
        return numpy.double(0.0),self.P0

    # overwritten in subclasses
    def get_x(self,t):
        pass

    # overwritten in subclasses
    def get_dxdt(self,t):
        pass

    def analytically_continue(self,t,checkpoint=None):
        """
        Analytically continue along the path to `t \in [0,1]`.
        """
        # use checkpoint, if provided. otherwise, find nearest checkpoint.
        if checkpoint is None:
            ti,Pi = self._nearest_checkpoint(t)
        else:
            ti,Pi = checkpoint

        df = self.df
        deg = self.deg
        xi,yi = Pi
        tip1 = t
        xip1 = self.get_x(tip1)

        # if the step from the checkpoint to desired t is zero then just return
        # the checkpoint
        if numpy.abs(xip1-xi) < 1e-15:
            return Pi

        # first determine if the y-root guesses are 'approximate solutions'. if
        # any of them are not then refine the step by analytically continuing
        # to an intermediate "time"
        for yij in yi:
            if smale_alpha(df,xip1,yij,deg) > smale_alpha0:
                tt = (ti+tip1)/2.0
                PP = self.analytically_continue(tt,checkpoint=(ti,Pi))
                return self.analytically_continue(tip1,checkpoint=(tt,PP))

        # next, determine if the approximate solutions will converge to
        # different associated solutions
        for j in xrange(deg):
            yij = yi[j]
            betaij = smale_beta(df,xip1,yij)
            for k in xrange(j+1,deg):
                yik = yi[k]
                betaik = smale_beta(df,xip1,yik)

                if numpy.abs(yij-yik) < 3*(betaij+betaik):  #XXX (was 2)
                    # approximate solutions don't lead to distinct
                    # roots. refine the step by analytically continuing to an
                    # intermedite time
                    tt = (ti+tip1)/2.0
                    PP = self.analytically_continue(tt,checkpoint=(ti,Pi))
                    return self.analytically_continue(tip1,checkpoint=(tt,PP))

        # finally, since we know that we have approximate solutions that will
        # converge to difference associated solutions we will Netwon iterate
        yip1 = tuple([ newton(df,xip1,yij) for yij in yi ])

        # return the t-point that we were able to step to as well as
        # the y-roots lying above that t-point
        return xip1, yip1

    def integrate(self,omega,x,y):
        """
        Integrates a differential `omega(x,y) = h(x,y)dx` along the path
        segment.
        """
        omega = sympy.lambdify((x,y),omega,'numpy')
        def integrand(ti):
            dxdt = self.get_dxdt(ti)
            xi,yi = self.analytically_continue(ti)
            return omega(xi,yi[0]) * dxdt

        # integrate the real an imaginary parts
        re = scipy.integrate.romberg(lambda t: integrand(t).real,0,1,
                                     tol=1e-14,rtol=1e-14,divmax=10)
        im = scipy.integrate.romberg(lambda t: integrand(t).imag,0,1,
                                     tol=1e-14,rtol=1e-14,divmax=10)
        return re + 1.0j*im

    def sample_points(self,tpts):
        """
        Return points along the curve at the provided (ordered) parameter
        values.
        """
        # if the starting t value is not zero, find the nearest checkpoint and
        # analytically continue to the starting t value from there
        N = len(tpts)
        tim1 = tpts[0]
        if numpy.abs(tpts[0]) < 1e-14:
            _,Pim1 = self._checkpoints[0]
        else:
            Pim1 = self.analytically_continue(tim1)

        sample = [Pim1]*N
        for i in range(1,N):
            # analytically continue to next point
            ti = tpts[i]
            Pi = self.analytically_continue(ti,checkpoint=(tim1,Pim1))
            sample[i] = Pi

            # update
            tim1 = ti
            Pim1 = Pi

        return sample

    def sample_uniform(self,N):
        """
        Return N points sampled uniformly along the parameterized curve.
        """
        tpts = numpy.linspace(0,1,N)
        return self.sample_points(tpts)

    def sample_clenshaw_curtis(self,N):
        """
        Return N points Chebyshev / Clenshaw-Curtis sampled along the
        parameterized curve.
        """
        theta = numpy.linspace(-numpy.pi,0,Npts)
        tpts = numpy.cos(theta)/2.0 + 0.5
        return self.sample(tpts)


class RiemannSurfacePathSegment_Line(RiemannSurfacePathSegment):
    def __init__(self,RS,P0,z0,z1,n_checkpoints=12):
        self.z0 = z0
        self.z1 = z1
        RiemannSurfacePathSegment.__init__(self,RS,P0)

    def get_x(self,t):
        return self.z0*(1-t) + self.z1*t

    def get_dxdt(self,t):
        return self.z1-self.z0


class RiemannSurfacePathSegment_Semicircle(RiemannSurfacePathSegment):
    def __init__(self,RS,P0,R,w,arg,dir,n_checkpoints=12):
        self.R = R
        self.w = w
        self.arg = arg
        self.dir = dir
        RiemannSurfacePathSegment.__init__(self,RS,P0)

    def get_x(self,t):
        return self.R * numpy.exp(1.0j*(self.dir*numpy.pi*t+self.arg)) + self.w

    def get_dxdt(self,t):
        return (self.R*1.0j*numpy.pi*self.dir) * \
            numpy.exp(1.0j*(self.dir*numpy.pi*t+self.arg))


class RiemannSurfacePath(object):
    """
    TODO: remove path_segment_data from constructor and replace with
    "automatic" path construction.
    """
    def __init__(self,RS,P0,P1=None,path_segment_data=None):
        """
        Construct a Riemann surface path.
        """
        # HACK. clean this up later. monodromy is the only function that calls
        # RiemannSurfacePath wtihout first constructing a RiemannSurface
        # object. Consider making monodromy inaccessible outside of
        # RiemannSurface. (mondromy and homology are internal, anyway...)
        if isinstance(RS,tuple):
            self.RS = RS
            self.f,self.x,self.y = RS
        else:
            self.RS = RS
            self.f = RS.f
            self.x = RS.x
            self.y = RS.y

        self.x0 = numpy.complex(P0[0])
        self.y0 = tuple(map(numpy.complex,P0[1]))
        self.P0 = (self.x0,self.y0)

        self.PathSegments = []
        self._initialize_segments(path_segment_data)

    def __str__(self):
        return r'RiemannSurfacePath defined on the curve %s == 0'%self.f

    def __call__(self,t,checkpoint=None):
        return self.analytically_continue(t,checkpoint=checkpoint)

    def _initialize_segments(self,path_segment_data):
        Pi = self.P0
        for datum in path_segment_data:
            # construct path segment from segment data
            if len(datum) == 2:
                z0,z1 = map(numpy.complex,datum)
                Segment = RiemannSurfacePathSegment_Line(self.RS,Pi,z0,z1)
            else:
                R,w,arg,dir = map(numpy.complex,datum)
                Segment = RiemannSurfacePathSegment_Semicircle(self.RS,Pi,
                                                               R,w,arg,dir)
            # analtyic continuation to the end of the segment is done at
            # construction so pick out the endpoint from the checkpoints
            ti,Pi = Segment._checkpoints[-1]
            self.PathSegments.append(Segment)

    def analytically_continue(self,t,checkpoint=None):
        if numpy.abs(t-1) < 1e-14:
            return self.PathSegments[-1]._checkpoints[-1][1]

        n = len(self.PathSegments)
        seg_index = int(numpy.floor(t*n))
        seg_t = t*n - seg_index
        Segment = self.PathSegments[seg_index]
        return Segment.analytically_continue(seg_t,checkpoint=checkpoint)

    def integrate(self,omega,x,y):
        """
        Integrate the differential `omega(x,y) = h(x,y)dx` along the path.
        """
        integral = numpy.complex(0.0)
        for Segment in self.PathSegments:
            integral += Segment.integrate(omega,x,y)
        return integral

    def sample_points(self,tpts_segment):
        """
        Return points along the curve at the provided (ordered) parameter
        values.
        """
        sample = []
        for Segment in self.PathSegments:
            sample.extend(Segment.sample_points(tpts_segment))
        return sample

    def sample_uniform(self,N):
        """
        Return N points sampled uniformly along the parameterized curve.
        """
        nSegs = len(self.PathSegments)
        n = N/nSegs
        tpts_segment = numpy.linspace(0,1,n)
        return self.sample_points(tpts_segment)

    def plot_x(self,N,**kwds):
        nSegs = len(self.PathSegments)
        ppseg = N #N/nSegs
        tpts_seg = numpy.linspace(0,1,ppseg)

        # create figure
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        # plot each segment
        eps = 0.02
        ctr = 0
        for k in range(nSegs):
            Segment = self.PathSegments[k]
            Pseg = Segment.sample_points(tpts_seg)
            xseg,yseg = zip(*Pseg)
            xseg = numpy.array(xseg,dtype=numpy.complex)

            for xx in xseg:
                if k <= nSegs/2:
                    ax.text(xx.real,xx.imag-eps,str(ctr),fontsize=9)
                else:
                    ax.text(xx.real,xx.imag+eps,str(ctr),fontsize=9)

                ctr += 1

            ax.plot(xseg.real,xseg.imag,'b')

        fig.show()


    def plot_y(self,N,**kwds):
        nSegs = len(self.PathSegments)
        ppseg = N #N/nSegs
        tpts_seg = numpy.linspace(0,1,ppseg)

        # create figure
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        colors_backup = ax._get_lines.color_cycle
        deg = len(self.y0)

        # plot each segment
        for k in range(nSegs):
            Segment = self.PathSegments[k]
            Pseg = Segment.sample_points(tpts_seg)
            xseg,yseg = zip(*Pseg)
            yseg = zip(*yseg)

            colors,colors_backup = tee(colors_backup)

            j = -1
            for yy in yseg:
                if k != 0: j = None
                else:      j += 1
                color = colors.next()
                yy = numpy.array(yy,dtype=numpy.complex)
                tseg = (tpts_seg+k)/nSegs
                ax.plot(tseg,yy.real,'-',tseg,yy.imag,'--',
                                 color=color,label=j,**kwds)

        ax.xaxis.set_ticks([numpy.double(k)/len(self.PathSegments)
                             for k in range(len(self.PathSegments)+1)])
        ax.grid(True,which='major')
        ax.legend(loc='best')

        fig.show()

    def plot_differential(self,omega,x,y,N):
        """
        Plots the differential `omega = omega(x,y) dx` on the path.

        NOTE: this can probably be written much more cleanly
        """
        nSegs = len(self.PathSegments)
        ppseg = N/nSegs
        tpts_seg = numpy.linspace(0,1,ppseg)

        oo = numpy.array([],dtype=numpy.complex)
        omega = sympy.lambdify((x,y),omega,'numpy')
        def parameterized_differential(ti,Segment=None):
            dxdt = Segment.get_dxdt(ti)
            xi,yi = Segment.analytically_continue(ti)
            return (xi,yi[0],omega(xi,yi[0]) * dxdt)
        pd = numpy.vectorize(parameterized_differential,excluded=['Segment'])

        # evaluate the differential along each segment
        xx = []
        yy = []
        tt = []
        oo = []
        for k in range(nSegs):
            Segment = self.PathSegments[k]
            xx_seg,yy_seg,oo_seg = pd(tpts_seg,Segment=Segment)
            xx.extend(xx_seg)
            yy.extend(yy_seg)
            tt.extend((tpts_seg+k)/nSegs)
            oo.extend(oo_seg)

        # get the x- and y-path interpolating points
        xx = numpy.array(xx,dtype=numpy.complex)
        yy = numpy.array(yy,dtype=numpy.complex)
        tt = numpy.array(tt,dtype=numpy.double)
        oo = numpy.array(oo,dtype=numpy.complex)

        # plotting: first axis is the x-path, second axis is the y-path, and
        # third axis is the value fo the differential along the path
        fig = plt.figure()
        ax1 = fig.add_subplot(1,3,1)
        ax2 = fig.add_subplot(1,3,2)
        ax3 = fig.add_subplot(1,3,3)

        ax1.plot(xx.real,xx.imag,'b.-')
        ax2.plot(yy.real,yy.imag,'g.-')
        ax3.plot(tt,oo.real,'r')
        ax3.plot(tt,oo.imag,'b')
        ax3.xaxis.set_ticks([numpy.double(k)/len(self.PathSegments)
                             for k in range(len(self.PathSegments)+1)])
        ax3.grid(True,which='major')

        fig.show()


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

    f = f9

    class DummyRiemannSurface(object):
        def __init__(self,f,x,y):
            self.f = f
            self.x = x
            self.y = y

    print "=== computing monodromy graph and base info ==="
    from abelfunctions.monodromy import monodromy_graph, show_paths
    G = monodromy_graph(f,x,y)
    base_point = G.node[0]['basepoint']
    base_sheets = G.node[0]['baselift']
    branch_points = [data['value'] for node,data in G.nodes(data=True)]

    print "=== showing path ==="
    show_paths(G)

    print "=== constructing path around branch point ==="
    bpt_index = 1
    print '\tbranch point: %s'%(bpt_index)
    print '\tconjugates:   %s'%(G.node[bpt_index]['conjugates'])
    path_segment_data = path_around_branch_point(G,bpt_index,1)
#    path_segment_data = path_around_infinity(G,-1)
    gamma = RiemannSurfacePath((f,x,y),(base_point,base_sheets),
                               path_segment_data=path_segment_data)

    print "=== fibre change ==="
    print "\toriginal\tnew\t"
    n = len(base_sheets)
    for k in range(n):
        print "\t%s\t%s"%(base_sheets[k],gamma(1.0)[1][k])

    gamma.plot_x(6)
    gamma.plot_y(32)

#     print "=== (computing holomorphic differentials)"
#     from abelfunctions.differentials import differentials
#     omega = differentials(f,x,y)[0]

#     print "omega =", omega

#     #
#     # path #0 construction
#     #
#     z0 = numpy.complex(-2)
#     z1 = numpy.complex(-1)
#     z2 = numpy.complex(-1.5 + 1.0j)
#     z3 = numpy.complex(-2.5 + 1.0j)
#     path_segment_data0 = [(z0,z1),(z1,z2),(z2,z3),(z3,z0)]
#     x0 = z0
#     y0 = polyroots(f,x,y,x0)
#     P0 = (x0,y0)
#     RS = DummyRiemannSurface(f,x,y)

#     print "\n=== (constructing path #0)"
#     gamma0 = RiemannSurfacePath(RS,P0,path_segment_data=path_segment_data0)

#     print "=== (plotting differential on path #0)"
#     gamma0.plot_differential(omega,x,y,N=64)

#     print "=== (integrating differential on path #0)"
#     integral = gamma0.integrate(omega,x,y)
#     print "\tvalue:", integral

#     pi = numpy.pi
#     R = numpy.double(1.0)
#     w = numpy.complex(-2)
#     path_segment_data1 = [
#         (R,w,0,1),
#         (R,w,pi,1),
#         ]
#     x0 = w+R
#     y0 = polyroots(f,x,y,x0)
#     P0 = (x0,y0)
#     RS = DummyRiemannSurface(f,x,y)

#     print "\n=== (constructing path #1)"
#     gamma1 = RiemannSurfacePath(RS,P0,path_segment_data=path_segment_data1)

#     print "=== (plotting differential on path #1)"
#     gamma1.plot_differential(omega,x,y,N=128)

#     print "=== (integrating differential on path #1)"
#     integral = gamma1.integrate(omega,x,y)
#     print "\tvalue:", integral
