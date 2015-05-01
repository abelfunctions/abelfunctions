r"""Riemann Surface Paths :mod:`abelfunctions.riemann_surface_path`
===============================================================

A framework for computing places along paths on Riemann surfaces.

The classes in this module follow the composite design pattern [1]_ with
:py:class:`RiemannSurfacePathPrimitive` acting as the "component" and
:py:class:`RiemannSurfacePath` acting as the "composite". The classes
:py:class:`RiemannSurfacePathLine` and :py:class:`RiemannSurfacePathArc`
define partricular types of paths.

Classes
-------

.. autosummary::

    RiemannSurfacePathPrimitive
    RiemannSurfacePath
    RiemannSurfacePathLine
    RiemannSurfacePathArc

References
----------

.. [1] E. Gamma, R. Helm, R. Johnson, J. Vlissides,
   *Design Patterns: Elements of Reusable Object-Oriented Software*,
   Pearson Education, 1994, pg. 163

Examples
--------

Contents
--------

"""

cimport cython
import numpy
cimport numpy
import sympy
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from .analytic_continuation cimport AnalyticContinuatorPuiseux
from .analytic_continuation_smale cimport AnalyticContinuatorSmale

cdef extern from 'math.h':
    int floor(double)
cdef extern from 'complex.h':
    complex cexp(complex)
    double cabs(complex)

cdef class RiemannSurfacePathPrimitive:
    r"""Primitive Riemann surface path object.

    Defines basic, primitive functionality for Riemann surface
    paths. Each path primitive is parameterized from :math:`t=0` to
    :math:`t=1`.

    Attributes
    ----------
    RS : RiemannSurface
        The Riemann surface on which this path primitive is defined.
    AC : AnalyticContinuator
        The mechanism this path uses for performing analytic
        continuation. An appropriate choice of analytic continuator, in
        part, depends on the proximity of the path to a singular point
        or branch point of the curve.
    x0 : complex
        Starting x-value of the path.
    y0 : complex[]
        Starting y-fibre of the path.
    segments

    Methods
    -------
    get_x
    get_dxdt
    analytically_continue
    get_y
    plot_x
    plot_y
    plot3d_x
    plot3d_y

    """
    property segments:
        r"""The individual :py:class:`RiemannSurfacePathPrimitive` objects that
        make up this object.

        Every `RiemannSurfacePathPrimitive` object contains a list of
        "path segments". In the case when this list of length one the
        list contains only the object itself.

        When the number of path segments is greater than one, the object
        should be coerced to a :py:class:`RiemannSurfacePath`
        object. Each element is a primitive representing an arc or
        straight line path in the complex x-plane.
        """
        def __get__(self):
            return self._segments

    property RS:
        def __get__(self):
            return self._RS

    property x0:
        def __get__(self):
            return self._x0

    property y0:
        def __get__(self):
            return numpy.array(self._y0, dtype=complex)

    property AC:
        def __get__(self):
            return self._AC

    def __init__(self, RiemannSurface RS, complex x0, complex[:] y0,
                 int ncheckpoints=8):
        r"""Intitialize the `RiemannSurfacePathPrimitive` using a
        `RiemannSurface`, `AnalyticContinuator`, and starting place.

        Note that the starting point must always be regular.
        """
        self._RS = RS
        self._x0 = x0
        self._y0 = y0
        self._segments = numpy.array([self], dtype=RiemannSurfacePathPrimitive)
        self._ncheckpoints = ncheckpoints
        if self._ncheckpoints > 0:
            self.set_analytic_continuator()
            self._initialize_checkpoints()
        self._str = None # cache the __str__ output

    def __str__(self):
        if self._str is None:
            self._set_str()
        return self._str

    def _set_str(self):
        is_cycle = False
        startx = numpy.complex(self._x0)
        endx = numpy.complex(self.get_x(1.0))

        # cheap check: if the x-endpoints match
        if numpy.abs(startx - endx) < 1e-12:
            starty = numpy.array(self._y0, dtype=complex)
            endy = numpy.array(self.get_y(1.0), dtype=complex)

            # expensive check: if the y-endpoints match
            if numpy.linalg.norm(starty - endy) < 1e-12:
                is_cycle = True

        if is_cycle:
            self._str = 'Cycle on the %s'%(self._RS.__repr__())
        else:
            self._str = self._set_str_noncycle()

    def _set_str_noncycle(self):
        r"""Generates the string representation of self in the case when self is
        not a cycle."""
        xstart = numpy.complex(self._x0)
        ystart = numpy.complex(self._y0[0])
        P = str((xstart, ystart))

        last_segment_AC = self.segments[-1].AC
        if isinstance(last_segment_AC, AnalyticContinuatorSmale):
            xend = numpy.complex(self.get_x(1.0))
            yend = numpy.complex(self.get_y(1.0)[0])
            Q = str((xend,yend))
        else:
            Q = str(last_segment_AC.target_place)
        return 'Path from %s to %s on the %s'%(P,Q,self._RS.__repr__())

    def set_analytic_continuator(self):
        r"""Select and appropriate analytic continuator for this path.

        If either the starting or ending place is at a discriminant
        point of the underlying curve then use an analytic continuation
        method that can distinguish between places lying above
        discriminant points. Otherwise, use a fast numerical method.

        Parameters
        ----------
        none

        Returns
        -------
        AnalyticContinuator
            An :class:`AnalyticContinuator` which defines how to
            analytically continue y-roots along a path.

        Notes
        -----
        Currently, this method will only look at the endpoint of the
        path to determine if it is epsilon close to a discriminant point
        of the curve. (Meaning that the places lying above the point can
        only be distinguished by Puiseux series.) A future update should
        determine if the path crosses through or passes near enough to
        the discriminant point.

        """
        # set the analytic continuator by checking if the end of the
        # path is close to a discriminant point
        x_end = self.get_x(1.0)
        b = self._RS.closest_discriminant_point(x_end, exact=True)
        if numpy.abs(numpy.complex(b) - x_end) < 1e-12:
            self._AC = AnalyticContinuatorPuiseux(self._RS, self, b)
        else:
            self._AC = AnalyticContinuatorSmale(self._RS, self)

    def __add__(self, RiemannSurfacePathPrimitive other):
        r"""Add two Riemann surface paths together.

        Checks if the ending place of `self` is equal to the ending
        place of `other`. If so, returns a `RiemannSurfacePath` object
        whose segments are a concatenation of the path segments of each
        summand.

        Parameters
        ----------
        other : RiemannSurfacePathPrimitive

        Returns
        -------
        RiemannSurfacePath
        """
        # try getting the segments of the other object. Doing so asserts
        # that the other object is of type RiemannSurfacePathPrimitve
        cdef RiemannSurfacePathPrimitive[:] segments
        try:
            segments = numpy.append(self.segments, other.segments)
        except AttributeError:
            raise TypeError('Summands must both be of '
                            'RiemannSurfacePathPrimitive type.')

        # assert that the endpoint of the segments of this path matches
        # with those of the other path
        cdef double eps = 1e-8
        cdef int deg = self.RS.deg
        cdef int k

        # get the ending place of the left RSPath (self) and the
        # starting place of the right RSPath (other).
        cdef RiemannSurfacePathPrimitive end_segment = self.segments[-1]
        cdef complex x_end = end_segment.get_x(1.0)
        cdef complex[:] y_end = end_segment.get_y(1.0)
        cdef complex x_start = other.x0
        cdef complex[:] y_start = other.y0

        # if the x- or y-values don't match, raise an error
        y_error = numpy.linalg.norm(numpy.array(y_start) - numpy.array(y_end))
        if (cabs(x_start - x_end) > eps) or (y_error > eps):
            raise ValueError('Cannot form sum of paths: starting place and '
                             'fibre of right Riemann surface path does not '
                             'match ending place of left path.')
        else:
            return RiemannSurfacePath(self.RS, self.x0, self.y0, segments)

    cdef int _nearest_checkpoint_index(self, double t):
        r"""Returns the index of the checkpoint closest to and preceding ``t``.

        Parameters
        ----------
        t : double

        Returns
        -------
        int
            The index ``k`` such that ``self._tcheckpoints[k] <= t`` but
            ``self._tcheckpoints[k+1] > t``.
        """
        cdef int n, k
        cdef double ti
        n = self._ncheckpoints
        for k in range(1,n):
            ti = self._tcheckpoints[k]
            if ti >= t:
                return k-1
        # use first checkpoint if something goes wrong
        return 0

    def _initialize_checkpoints(self):
        r"""Analytically continue along the entire path recording y-values at
        evenly spaced points.

        We cache the y-values at various evenly-spaced points :math:`t
        \in [0,1]` so one doesn't have to analytically continue from
        :math:`t=0` every time.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        cdef double[:] t
        cdef complex[:] x
        cdef complex[:,:] y
        cdef int n,k
        cdef double tim1,ti
        cdef complex xim1,xi
        cdef complex[:] yim1,yi

        n = self._ncheckpoints
#        tend = 1. - 1./(n+1)
        tend = 1
        t = numpy.linspace(0, tend, n)
        x = numpy.array([self.get_x(ti) for ti in t], dtype=complex)
        y = numpy.zeros((n, self._RS.deg), dtype=complex)

        # for each t-checkpoint compute the corresponding x- and
        # y-checkpoint by analytically continuing.
        tim1 = 0.0
        xim1 = self._x0
        yim1 = self._y0
        y[0,:] = yim1
        for i in range(1,n):
            ti = t[i]
            xi = self.get_x(ti)
            yi = self.analytically_continue(xim1, yim1, xi)
            y[i,:] = yi
            xim1 = xi
            yim1 = yi

        self._tcheckpoints = t
        self._xcheckpoints = x
        self._ycheckpoints = y

    cpdef complex get_x(self, double t):
        r"""Return the x-part of the path at :math:`t \in [0,1]`.

        Parameters
        ----------
        t : double

        Returns
        -------
        complex

        """
        raise NotImplementedError('Must override RiemannSurfacePathPrimitive.'
                                  'get_x() method in subclass.')

    cpdef complex get_dxdt(self, double t):
        r"""Return the derivative of the x-part of the path at :math:`t \in
        [0,1]`.

        Parameters
        ----------
        t : double

        Returns
        -------
        complex

        """
        raise NotImplementedError('Must override RiemannSurfacePathPrimitive.'
                                  'get_dxdt() method in subclass.')

    cpdef complex[:] analytically_continue(self, complex xi, complex[:] yi,
                                           complex xip1):
        r"""Analytically continue the fibre ``yi`` from ``xi`` to ``xip1``.

        .. note::

           Calls internal AnalyticContinuator.

        Parameters
        ----------
        xi : complex
        yi : complex[]
            The current x,y-fibre pair.
        xip1 : complex
            The target complex x-point.

        Returns
        -------
        complex[]
            The fibre above ``xip1``.

        """
        cdef complex[:] yip1 = self._AC.analytically_continue(xi, yi, xip1)
        return yip1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex[:] get_y(self, double t):
        r"""Return the y-fibre of the path at :math:`t \in [0,1]`.

        Parameters
        ----------
        t : double

        Returns
        -------
        complex[:]

        """
        cdef int n, k
        cdef double tim1
        cdef complex xim1, xi
        cdef complex[:] yim1, yi

        # get the closest checkpoint to the desired t-value
        i = self._nearest_checkpoint_index(t)
        tim1 = self._tcheckpoints[i]
        xim1 = self._xcheckpoints[i]
        yim1 = self._ycheckpoints[i]

        # analytically continue to target
        xi = self.get_x(t)
        yi = self.analytically_continue(xim1, yim1, xi)
        return yi

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex integrate(self, Differential omega):
        r"""Integrate `omega` on the path using its analytic continuator.

        Delegates integration to the path's
        :class:`AnalyticContinuator`. The strategy for analytic
        continuation depends on if this path terminates at a
        discriminant point of the curve.

        Parameters
        ----------
        omega : Differential

        Returns
        -------
        complex
        """
        return self.AC.integrate(omega)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex[:] evaluate(self, Differential omega, double[:] t):
        r"""Evaluates `omega` along the path at `t` between 0 and 1.

        Parameters
        ----------
        omega : Differential
        t : double[:]
            An array of `t` between 0 and 1.

        Returns
        -------
        complex[:]
            The differential omega evaluated along the path at each of
            the points in `t`.
        """
        omega_gamma = self.AC.parameterize(omega)
        return omega_gamma(t)

    def plot_x(self, N=128, t0=0, t1=1, **kwds):
        r"""Plot the x-part of the path in the complex x-plane.

        Additional arguments and keywords are passed to
        ``matplotlib.pyplot.plot``.

        Parameters
        ----------
        N : int
            The number of interpolating points used to plot.
        t0 : double
            Starting t-value in [0,1].
        t1 : double
            Ending t-value in [0,1].

        Returns
        -------
        matplotlib lines array.

        """
        t = numpy.linspace(t0,t1,N)
        x = numpy.array([self.get_x(ti) for ti in t],
                        dtype=numpy.complex)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        lines, = ax.plot(x.real, x.imag, **kwds)
        return fig

    def plot_y(self, N=128, t0=0, t1=1, **kwds):
        r"""Plot the y-part of the path in the complex y-plane.

        Additional arguments and keywords are passed to
        ``matplotlib.pyplot.plot``.

        Parameters
        ----------
        N : int
            The number of interpolating points used to plot.
        t0 : double
            Starting t-value in [0,1].
        t1 : double
            Ending t-value in [0,1].

        Returns
        -------
        matplotlib lines array.

        """
        t = numpy.linspace(t0,t1,N)
        y = numpy.array([self.get_y(ti)[0] for ti in t],
                        dtype=numpy.complex)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        lines, = ax.plot(y.real, y.imag, **kwds)
        return fig

    def plot3d_x(self, N=128, t0=0, t1=1, **kwds):
        r"""Plot the x-part of the path in the complex x-plane with the
        parameter :math:`t \in [0,1]` along the perpendicular axis.

        Additional arguments and keywords are passed to
        ``matplotlib.pyplot.plot``.

        Parameters
        ----------
        N : int
            The number of interpolating points used to plot.
        t0 : double
            Starting t-value in [0,1].
        t1 : double
            Ending t-value in [0,1].


        Returns
        -------
        matplotlib lines array.

        """
        z = numpy.zeros(N)
        t = numpy.linspace(t0,t1,N)
        y = numpy.array([self.get_x(ti) for ti in t],
                        dtype=numpy.complex)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection='3d')
        ax.plot(y.real, y.imag, t, **kwds)

        # draw a grey "shadow" below the plot
        try:
            kwds.pop('color')
        except:
            pass
        kwds['alpha'] = 0.5
        ax.plot(y.real, y.imag, z, color='grey', **kwds)
        return fig

    def plot3d_y(self, N=128, t0=0, t1=1, **kwds):
        r"""Plot the y-part of the path in the complex y-plane with the
        parameter :math:`t \in [0,1]` along the perpendicular axis.

        Additional arguments and keywords are passed to
        ``matplotlib.pyplot.plot``.

        Parameters
        ----------
        N : int
            The number of interpolating points used to plot.
        t0 : double
            Starting t-value in [0,1].
        t1 : double
            Ending t-value in [0,1].

        Returns
        -------
        matplotlib lines array.

        """
        z = numpy.zeros(N)
        t = numpy.linspace(t0,t1,N)
        y = numpy.array([self.get_y(ti)[0] for ti in t],
                        dtype=numpy.complex)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection='3d')
        ax.plot(y.real, y.imag, t, **kwds)

        # draw a grey "shadow" below the plot
        try:
            kwds.pop('color')
        except:
            pass
        kwds['alpha'] = 0.5
        ax.plot(y.real, y.imag, z, color='grey', **kwds)
        return fig


cdef class RiemannSurfacePathLine(RiemannSurfacePathPrimitive):
    r"""A Riemann surface path for which the x-part of the path is a line
    segment.

    Attributes
    ----------
    z0,z1 : complex
       The starting and ending points of the complex x-line.

    """
    def __init__(self, RiemannSurface RS, complex x0, complex[:] y0,
                 complex z0, complex z1, int ncheckpoints=8):
        self.z0 = z0
        self.z1 = z1
        RiemannSurfacePathPrimitive.__init__(
            self, RS, x0, y0, ncheckpoints=ncheckpoints)

    def __repr__(self):
        return 'Line(%s,%s)'%(self.z0,self.z1)

    cpdef complex get_x(self, double t):
        return self.z0*(1 - t) + self.z1*t

    cpdef complex get_dxdt(self, double t):
        return self.z1 - self.z0


cdef class RiemannSurfacePathArc(RiemannSurfacePathPrimitive):
    r"""A Riemann surface path for which the x-part of the path is an arc.

    Attributes
    ----------
    R : complex
        The radius of the semicircle. (Complex type for coercion
        performance.)
    w : complex
        The center of the semicircle.
    theta : complex
        The starting angle (in radians) on the semicircle. Usually 0 or
        :math:`\pi`.  (Complex type for coercion performance.)
    dtheta : complex
        The number of radians to travel where the sign of `dtheta`
        indicates direction. The absolute value of `dtheta` is equal to
        the arc length.

    """
    def __init__(self, RiemannSurface RS, complex x0, complex[:] y0,
                 complex R, complex w, complex theta, complex dtheta,
                 int ncheckpoints=8):
        self.R = R
        self.w = w
        self.theta = theta
        self.dtheta = dtheta
        RiemannSurfacePathPrimitive.__init__(
            self, RS, x0, y0, ncheckpoints=ncheckpoints)

    def __repr__(self):
        return 'Arc(%s,%s,%s,%s)'%(self.R,self.w,self.theta,self.dtheta)

    cpdef complex get_x(self, double t):
        return self.R*cexp(1.0j*(self.theta + t*self.dtheta)) + \
            self.w

    cpdef complex get_dxdt(self, double t):
        return (self.R*1.0j*self.dtheta) * \
            cexp(1.0j*(self.theta + t*self.dtheta))


cdef class RiemannSurfacePathRay(RiemannSurfacePathPrimitive):
    r"""A Riemann surface path for which the x-part goes to infinity.

    Given a starting point :math:`x_0` the x-path :math:`\gamma_x :
    [0,1] \to \mathbb{C}_x` going to infinity is the one that travels
    radially outward from the origin :math:`x=0` given by the equation

    .. math::

        \gamma_x(t) = \frac{x_0}{1-t}

    """
    def __init__(self, RiemannSurface RS, complex x0, complex[:] y0,
                 int ncheckpoints=8):
        RiemannSurfacePathPrimitive.__init__(
            self, RS, x0, y0, ncheckpoints=ncheckpoints)

    def __repr__(self):
        return 'Ray(%s)'%(self.x0)

    def set_analytic_continuator(self):
        self._AC = AnalyticContinuatorPuiseux(self._RS, self, sympy.oo)

    cpdef complex get_x(self, double t):
        if t == 1.0: t -= 1e-12
        return self._x0 / (1 - t)

    cpdef complex get_dxdt(self, double t):
        if t == 1.0: t -= 1e-12
        return -self._x0 / (1 - t)**2


cdef class RiemannSurfacePath(RiemannSurfacePathPrimitive):
    r"""A continuous, piecewise differentiable path on a Riemann surface.

    RiemannSurfacePath is a composite of
    :py:class:`RiemannSurfacePathPrimitive` objects. This path is
    parameterized for :math:`t \in [0,1]`.

    Methods
    -------
    get_x
    get_dxdt
    get_y

    """
    def __init__(self, RiemannSurface RS, complex x0, complex[:] y0,
                 RiemannSurfacePathPrimitive[:] segments):
        # RiemannSurfacePath delegates all analytic continuation to each
        # of its components, so we intialize its parent with a null
        # AnalyticContinuator object.
        #
        # Additionally, setting ncheckpoints to "0" prevents
        # self._initialize_checkpoints() from executing, which only
        # makes sense on a single path segment / path primitive.
        RiemannSurfacePathPrimitive.__init__(self, RS, x0, y0, ncheckpoints=0)

        # important: self._segments must be set after the parent
        # intialization call since parent will set self._segments equal
        # to "self"
        self._segments = segments
        self._nsegments = len(segments)

    def __repr__(self):
        s = ','.join(segment.__repr__() for segment in self._segments)
        return 'RiemannSurfacePath(' + s + ') on the ' + self.RS.__repr__()

    cdef int _get_segment_index(self, double t):
        r"""Returns the index of the path segment located at the given :math:`t
        \in [0,1]`.

        .. note::

            This routine computes the appropriate path segment index
            without resorting to branching. Such an approach is needed
            since :math:`t = 1.0` should return the index of the final
            segment.

        Parameters
        ----------
        t : double

        Returns
        -------
        int
            An integer between ``0`` and ``self._nsegments-1``
            representing the index of the segment on which ``t`` lies.

        """
        cdef int k = floor(t*self._nsegments)
        cdef int diff = (self._nsegments - 1) - k
        cdef int dsgn = diff >> 31
        return k + (diff & dsgn)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex get_x(self, double t):
        r"""Return the x-part of the path at :math:`t \in [0,1]`.

        Parameters
        ----------
        t : double

        Returns
        -------
        complex

        Notes
        -----
        This RiemannSurfacePath is parameterized for :math:`t \in
        [0,1]`. However, internally, each segment is separately
        parameterized for :math:`t \in [0,1]`. This routine performs an
        appropriate scaling.

        """
        cdef RiemannSurfacePathPrimitive seg_k
        cdef complex x
        cdef int k = self._get_segment_index(t)
        cdef double t_seg = t*self._nsegments - k
        seg_k = self._segments[k]
        x = seg_k.get_x(t_seg)
        return x

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex get_dxdt(self, double t):
        r"""Return the derivative of the x-part of the path at :math:`t \in
        [0,1]`.

        Parameters
        ----------
        t : double

        Returns
        -------
        complex

        Notes
        -----
        This RiemannSurfacePath is parameterized for :math:`t \in
        [0,1]`. However, internally, each segment is separately
        parameterized for :math:`t \in [0,1]`. This routine
        performs an appropriate scaling.

        .. warning::

           Riemann surface paths are only piecewise differentiable and
           therefore may have discontinuous derivatives at the
           boundaries. Therefore, it may be more useful to perform
           segment-wise operations instead of operations on the whole of
           this object.


        """
        cdef RiemannSurfacePathPrimitive seg_k
        cdef complex dxdt
        cdef int k = self._get_segment_index(t)
        cdef double t_seg = t*self._nsegments - k
        seg_k = self._segments[k]
        dxdt = seg_k.get_dxdt(t_seg)
        return dxdt

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex[:] get_y(self, double t):
        r"""Return the y-fibre of the path at :math:`t \in [0,1]`.

        Parameters
        ----------
        t : double

        Returns
        -------
        complex[:]

        Notes
        -----
        This RiemannSurfacePath is parameterized for
        :math:`t \in [0,1]`. However, internally, each segment is
        separately parameterized for :math:`t \in [0,1]`. This routine
        performs an appropriate scaling.

        """
        cdef RiemannSurfacePathPrimitive seg_k
        cdef complex[:] y
        cdef int k = self._get_segment_index(t)
        cdef double t_seg = t*self._nsegments - k
        seg_k = self._segments[k]
        y = seg_k.get_y(t_seg)
        return y

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex[:] evaluate(self, Differential omega, double[:] t):
        r"""Evaluates `omega` along the path at `N` uniform points.

        .. todo::

            Note: right now it doesn't matter what the values in `t`
            are. This function will simply turn `t` into a bunch of
            uniformly distributed points between 0 and 1.

        Parameters
        ----------
        omega : Differential
        t : double[:]
            An array of `t` between 0 and 1.

        Returns
        -------
        complex[:]
            The differential omega evaluated along the path at `N` points.
        """
        cdef RiemannSurfacePathPrimitive segment
        cdef int N = len(t)
        cdef int nsegs = len(self.segments)
        cdef int ppseg = int(N/nsegs)
        cdef complex[:] values = numpy.zeros(nsegs*ppseg,dtype=complex)
        cdef complex[:] values_seg = numpy.zeros(ppseg,dtype=complex)
        cdef double[:] tseg = numpy.linspace(0,1,ppseg)
        cdef int j,k

        for k in range(nsegs):
            segment = self.segments[k]
            values_seg = segment.evaluate(omega,tseg)
            for j in range(ppseg):
                values[k*ppseg+j] = values_seg[j]
        return values

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex integrate(self, Differential omega):
        r"""Integrate `omega` on the path using its analytic continuator.

        Delegates integration to the path's
        :class:`AnalyticContinuator`. The strategy for analytic
        continuation depends on if this path terminates at a
        discriminant point of the curve.

        Parameters
        ----------
        omega : Differential

        Returns
        -------
        complex
        """
        cdef RiemannSurfacePathPrimitive segment
        cdef complex integral = 0.0
        for segment in self.segments:
            integral += segment.integrate(omega)
        return integral
