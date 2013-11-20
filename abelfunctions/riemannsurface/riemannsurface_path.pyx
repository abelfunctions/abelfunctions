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
cimport numpy
import scipy
import sympy
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

from itertools import tee
from smale cimport *

import abelfunctions


# def factorial(n):
#     return reduce(lambda a,b: a*b, xrange(1,n+1))

# cpdef newton(df, complex xip1, complex yij):
#     cdef double eps = 1e-14
#     cdef complex step = 1, df1 = 0.0

#     while numpy.abs(step) > eps:
#         # check if Df is invertible. (If not, then we are at a
#         # critical point.)
#         df1 = <complex>df[1](xip1,yij)
#         if numpy.abs(df1) < eps:
#             return yij

#         # Newton iterate
#         step = df[0](xip1,yij)/df1
#         yij -= step
#     return yij

# def smale_beta(df,xip1,yij):
#     return numpy.abs( df[0](xip1,yij)/df[1](xip1,yij) )

# def smale_gamma(df,xip1,yij,deg):
#     df1 = df[1](xip1,yij)
#     bounds = ( numpy.abs(df[k](xip1,yij)/(factorial(k)*df1))**(1./(k-1.0))
#                for k in xrange(2,deg+1) )
#     return max(bounds)

# smale_alpha0 = numpy.double(13.0 - 2.0*numpy.sqrt(17.0))/4.0
# def smale_alpha(df,xip1,yij,deg):
#     return smale_beta(df,xip1,yij) * smale_gamma(df,xip1,yij,deg)

smale_alpha0 = numpy.double(13.0 - 2.0*numpy.sqrt(17.0))/4.0


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

#    return sympy.mpmath.polyroots(coeffs)


def path_segments_from_cycle(cycle, G, base_point=None):
    """
    Given a cycle, which is a list of the form

        (...,s_i,(b_i,n_i),....)

    where s_i is a sheet index, b_i is a branch point, and n_i is the
    number of times and direction one goes around the branch point,
    return a list of path segments parameterizing the input cycle.

    The path segment is constructed by performing repeated calls to
    path_around_granch_point() and path_around_infinity().

    Input:

    * cycle: a cycle in the form as output by homology()

    * G: the monodromy graph, as output by monodromy()

    * base_point: (optional) a custom base point
    """
    path_segments = []

    # for each (branch point, rotation number) pair appearing in the
    # cycle, determine the path segments in the complex x-plane going
    # around that branch point "rotation number" number of times
    branch_points = [data['value'] for key,data in G.nodes(data=True)]
    for (bpt,rot) in cycle[1::2]:
        if (bpt == sympy.oo) or (bpt == numpy.Inf):
            bpt_path_seg = path_around_infinity(G,rot)
        else:
            idx = branch_points.index(bpt)
            bpt_path_seg = path_around_branch_point(G,idx,rot)
        path_segments.extend(bpt_path_seg)

    # if a custom base point is provided then add a line segment going
    # from the custom base point to the default one chosen by monodromy
    if base_point:
        seg = (base_point, G.node[0]['basepoint'])
        path_segments = [seg] + path_segments + [tuple(reversed(seg))]

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
    dir = -1 if rot > 0 else 1   #XXX

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
    def __init__(self,RS,P0,n_checkpoints=8):
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
        # monodromy calls RiemannSurfacePath without first constructing
        # a RiemannSurface object
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
            if smale_alpha(df,xip1,yij) > smale_alpha0:
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

    def integrate(self,omega,x,y,**kwds):
        """
        Integrates a differential `omega(x,y) = h(x,y)dx` along the path
        segment.
        """
        omega = sympy.lambdify((x,y),omega,'numpy')
        def integrand(ti):
            dxdt = self.get_dxdt(ti)
            xi,yi = self.analytically_continue(ti)
            return omega(xi,yi[0]) * dxdt

#         # integrate the real an imaginary parts
#         re = scipy.integrate.romberg(lambda t: integrand(t).real,0,1,
#                                      **kwds)
#         im = scipy.integrate.romberg(lambda t: integrand(t).imag,0,1,
#                                      **kwds)
#         return re + 1.0j*im
        return scipy.integrate.romberg(integrand,0,1,**kwds)


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
        theta = numpy.linspace(-numpy.pi,0,N)
        tpts = numpy.cos(theta)/2.0 + 0.5
        return self.sample(tpts)


class RiemannSurfacePathSegment_Line(RiemannSurfacePathSegment):
    def __init__(self,RS,P0,z0,z1,n_checkpoints=5):
        self.z0 = z0
        self.z1 = z1
        RiemannSurfacePathSegment.__init__(self,RS,P0,
                                           n_checkpoints=n_checkpoints)

    def get_x(self,t):
        return self.z0*(1-t) + self.z1*t

    def get_dxdt(self,t):
        return self.z1-self.z0


class RiemannSurfacePathSegment_Semicircle(RiemannSurfacePathSegment):
    def __init__(self,RS,P0,R,w,arg,dir,n_checkpoints=9):
        self.R = R
        self.w = w
        self.arg = arg
        self.dir = dir
        RiemannSurfacePathSegment.__init__(self,RS,P0,
                                           n_checkpoints=n_checkpoints)

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
    def __init__(self, RS, P0, P1=None, path_segments=None, cycle=None):
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

        # for convenience, if a cycle is provided instead of path segments
        # then the path segments are computed
        if cycle:
            G = abelfunctions.monodromy_graph(self.f,self.x,self.y)
            path_segments = path_segments_from_cycle(cycle,G,base_point=P0[0])

        # initialize the path segemnts with checkpoints by anaytically
        # continuing
        self.PathSegments = []
        self._initialize_segments(path_segments)

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

    def integrate(self,omega,x,y,**kwds):
        """
        Integrate the differential `omega(x,y) = h(x,y)dx` along the path.
        """
        integral = numpy.complex(0.0)
        for Segment in self.PathSegments:
            integral += Segment.integrate(omega,x,y,**kwds)
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
        ppseg = int(N/nSegs)
        tpts_seg = numpy.linspace(0,1,ppseg)

        # create figure
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)

        # plot each segment
        eps = 0.02
        ctr = 0
        for k in range(nSegs):
            Segment = self.PathSegments[k]
            Pseg = Segment.sample_points(tpts_seg)
            xseg,yseg = zip(*Pseg)
            xseg = numpy.array(xseg,dtype=numpy.complex)
            dxseg = numpy.array([Segment.get_dxdt(ti) for ti in tpts_seg],
                                dtype=numpy.complex)
            tseg = (tpts_seg+k)/nSegs

            for xx in xseg:
                if k <= nSegs/2:
                    ax1.text(xx.real,xx.imag-eps,str(ctr),fontsize=9)
                else:
                    ax1.text(xx.real,xx.imag+eps,str(ctr),fontsize=9)

                ctr += 1

            ax1.plot(xseg.real,xseg.imag,'b')
            ax2.plot(tseg,dxseg.real,'b',tseg,dxseg.imag,'b--')

        fig.show()

    def plot_y(self,N,**kwds):
        nSegs = len(self.PathSegments)
        ppseg = N/nSegs
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
        xc = []
        yc = []
        for k in range(nSegs):
            Segment = self.PathSegments[k]

            xx_seg,yy_seg,oo_seg = pd(tpts_seg,Segment=Segment)
            xx.extend(xx_seg)
            yy.extend(yy_seg)
            tt.extend((tpts_seg+k)/nSegs)
            oo.extend(oo_seg)

            # checkpoint values
            ti,pc = zip(*Segment._checkpoints)
            xck,ycallk = zip(*pc)
            yck = zip(*ycallk)[0]
            xc.extend(xck)
            yc.extend(yck)

        # get the x- and y-path interpolating points
        xx = numpy.array(xx,dtype=numpy.complex)
        yy = numpy.array(yy,dtype=numpy.complex)
        tt = numpy.array(tt,dtype=numpy.double)
        oo = numpy.array(oo,dtype=numpy.complex)
        xc = numpy.array(xc,dtype=numpy.complex)
        yc = numpy.array(yc,dtype=numpy.complex)

        # plotting: first axis is the x-path, second axis is the y-path, and
        # third axis is the value fo the differential along the path
        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,3)
        ax3 = fig.add_subplot(1,2,2)

        ax1.plot(xx.real,xx.imag,'b-',xc.real,xc.imag,'bo')
        ax2.plot(yy.real,yy.imag,'g-',yc.real,yc.imag,'go')
        ax3.plot(tt,oo.real,'b-')
        ax3.plot(tt,oo.imag,'b--')
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

    f = f10

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

#     print "=== constructing path around branch point ==="
#     bpt_index = 5
#     print '\tbranch point: %s'%(bpt_index)
#     print '\tconjugates:   %s'%(G.node[bpt_index]['conjugates'])
# #    path_segment_data = path_around_branch_point(G,bpt_index,1)
#     path_segment_data = path_around_infinity(G,-3)
#     gamma = RiemannSurfacePath((f,x,y),(base_point,base_sheets),
#                                path_segment_data=path_segment_data)

    print "=== computing homology and c-cycles ==="
    from abelfunctions.homology import homology
    cycles = homology(f,x,y,verbose=True)['c-cycles']
    gammas = [RiemannSurfacePath((f,x,y),(base_point,base_sheets),
                                 cycle=cycle)
              for cycle in cycles]

    print "=== (computing holomorphic differentials)"
    from abelfunctions.differentials import differentials
    omegas = differentials(f,x,y)
    for omega in omegas:
        sympy.pprint(omega)
