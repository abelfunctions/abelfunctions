"""
Riemann Surface Paths
=====================

The classes in this module follow the Composite design pattern [1]_ with
:class:RiemannSurfacePathPrimitive acting as the "component" and
:class:RiemannSurfacePath acting as the "composite". The classes
:class:RiemannSurfacePathLine and :class:RiemannSurfacePathCircle define
partricular types of paths.

References
----------

.. [1] E. Gamma, R. Helm, R. Johnson, J. Vlissides, *Design
Patterns: Elements of Reusable Object-Oriented Software*,
Pearson Education, 1994, pg. 163


"""

cimport cython
import numpy
cimport numpy
import sympy
import matplotlib
import matplotlib.pyplot as plt

cdef extern from 'math.h':
    int floor(double)
cdef extern from 'complex.h':
    complex cexp(complex)
    double cabs(complex)


cdef class RiemannSurfacePathPrimitive:
    """Primitive Riemann surface path object.

    Defines basic, primitive functionality for Riemann surface
    paths. Each path primitive is parameterized from t=0 to t=1.

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
    y0 : complex[:]
        Starting y-fibre of the path.
    segments : RiemannSurfacePathSegments

    Methods
    -------
    get_x(double t)
        Return the x-part of the path at `t` for `t \in [0,1]`.
    get_dxdt(double t)
        Return the deriviatve of the x-path at `t` for `t \in [0,1]`.
    get_y(double t)
        Return the y-fibre of the path at `t` for `t \in [0,1]`.
    plot(double[:] t, *args, **kwds)
        Plots the path in the complex x- and y-planes at the t-points
        given. *args and **kwds are passed to matplotlib.pyplot.plot.
    """
    def __init__(self, RiemannSurface RS, AnalyticContinuator AC,
                 complex x0, complex[:] y0, int ncheckpoints=6):
        print '--- RSPP __init__()'
        self.RS = RS
        self.x0 = x0
        self.y0 = y0
        self.AC = AC
        self.segments = numpy.array([self], dtype=RiemannSurfacePathPrimitive)
        self._ncheckpoints = ncheckpoints
        self._initialize_checkpoints()
        print '--- RSPP __init__() END'

    def __add__(self, other):
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
        #
        # XXX do we have to check the entire fibre or just the first
        # fibre component?
        cdef double eps = 1e-8
        cdef RiemannSurfacePathPrimitive end_segment = self.segments[-1]
        cdef complex x_end = end_segment.get_x(1)
        cdef complex[:] y_end = end_segment.get_y(1)
        cdef complex x_start = other.x0
        cdef complex[:] y_start = other.y0
        cdef int deg = self.RS.deg
        cdef int k

        # compute the 1-norm of the difference of the y-values
        cdef double norm = 0
        for k in range(deg):
            norm += cabs(y_start[k] - y_end[k])

        # if the x- or y-values don't match, raise an error
        if (cabs(x_start - x_end) > eps) or (norm > eps):
            raise ValueError('Cannot form sum of paths: starting place and '
                             'fibre of right Riemann surface path does not '
                             'match ending place of left path.')
        else:
            return RiemannSurfacePath(self.RS, self.x0, self.y0, segments)

    cdef int _nearest_checkpoint_index(self, double t):
        """Returns the index of the checkpoint closest to and preceding `t`."""
        cdef int n,k
        cdef double ti

        n = self._ncheckpoints
        for k in range(1,n):
            ti = self._tcheckpoints[k]
            if ti >= t:
                return k-1

        # use base point if a valid checkpoint isn't found
        return 0

    def _initialize_checkpoints(self):
        """Analytically continue along the entire path recording y-values at
        evenly spaced points.

        We cache the y-values at various evenly-spaced points `t \in
        [0,1]` so one doesn't have to analytically continue from `t=0`
        every time.
        """
        cdef double[:] t
        cdef complex[:] x
        cdef complex[:,:] y
        cdef int n,k
        cdef double tim1,ti
        cdef complex xim1,xi
        cdef complex[:] yim1,yi

        n = self._ncheckpoints
        t = numpy.linspace(0, 1, n)
        x = numpy.array([self.get_x(ti) for ti in t], dtype=complex)
        y = numpy.zeros((n,self.RS.deg), dtype=complex)

        tim1 = 0.0
        xim1 = self.x0
        yim1 = self.y0
        for k in range(1,n):
            ti = t[k]
            xi = self.get_x(ti)
            yi = self.analytically_continue(xim1, yim1, xi)

            y[k] = yi
            xim1 = xi
            yim1 = yi

        self._tcheckpoints = t
        self._xcheckpoints = x
        self._ycheckpoints = y

    cpdef complex get_x(self, double t):
        """Return the x-part of the path at `t \in [0,1]`."""
        raise NotImplementedError('Must override RiemannSurfacePathPrimitive.'
                                  'get_x() method in subclass.')

    cpdef complex get_dxdt(self, double t):
        """Return the derivative of the x-part of the path at `t \in [0,1]`."""
        raise NotImplementedError('Must override RiemannSurfacePathPrimitive.'
                                  'get_dxdt() method in subclass.')

    cpdef complex[:] analytically_continue(self, complex xi, complex[:] yi,
                                           complex xip1):
        """Analytically continue the fibre `yi` from `xi` to `xip1`.

        .. note::

           Calls internal AnalyticContinuator.
        """
        cdef complex[:] yip1 = self.AC.analytically_continue(
            self, xi, yi, xip1)
        return yip1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex[:] get_y(self, double t):
        """Return the y-fibre of the path at `t \in [0,1]`."""
        cdef int n,k
        cdef double tim1
        cdef complex xim1,xi
        cdef complex[:] yim1,yi

        # get the closest checkpoint to the desired t-value
        k = self._nearest_checkpoint_index(t)
        tim1 = self._tcheckpoints[k]
        xim1 = self._xcheckpoints[k]
        yim1 = self._ycheckpoints[k]

        # analytically continue to target
        xi = self.get_x(t)
        yi = self.analytically_continue(xim1, yim1, xi)
        return yi

    def plot_x(self, double[:] t, *args, **kwds):
        """Plot the x-part of the path in the complex x-plane.

        Arguments
        ---------
        t : double[:]
            List of `t \in [0,1]` values at which to plot `x=x(t)`.
        *args, **kwds
            Additional arguments and keywords are passed to
            matplotlib.pyplot.plot.

        Returns
        -------
        matplotlib lines array.
        """
        x = numpy.array([self.get_x(ti) for ti in t],
                        dtype=numpy.complex)
        p = plt.plot(x.real, x.imag, *args, **kwds)
        return p

    def plot_y(self, double[:] t, *args, **kwds):
        """Plot the y-part of the path in the complex y-plane.

        Arguments
        ---------
        t : double[:]
            List of `t \in [0,1]` values at which to plot `y=y(t)`.
        *args, **kwds
            Additional arguments and keywords are passed to
            matplotlib.pyplot.plot.

        Returns
        -------
        matplotlib lines array.
        """
        y = numpy.array([self.get_y(ti)[0] for ti in t],
                        dtype=numpy.complex)
        p = plt.plot(y.real, y.imag, *args, **kwds)
        return p


cdef class RiemannSurfacePathLine(RiemannSurfacePathPrimitive):
    """A Riemann surface path for which the x-part of the path is a line
    segment.

    Attributes
    ----------
    z0,z1 : complex
       The starting and ending points of the complex x-line.
    """
    def __init__(self, RiemannSurface RS, AnalyticContinuator AC,
                 complex x0, complex[:] y0, complex z0, complex z1,
                 int ncheckpoints=5):
        self.z0 = z0
        self.z1 = z1
        RiemannSurfacePathPrimitive.__init__(self, RS, AC, x0, y0,
                                             ncheckpoints=ncheckpoints)

    def __str__(self):
        return 'RiemannSurfacePathLine:\nstart: %s\nend:  %s'%(self.z0,self.z1)

    cpdef complex get_x(self, double t):
        return self.z0*(1 - t) + self.z1*t

    cpdef complex get_dxdt(self, double t):
        return self.z1 - self.z0


cdef class RiemannSurfacePathArc(RiemannSurfacePathPrimitive):
    """A Riemann surface path for which the x-part of the path is an arc.

    Attributes
    ----------
    R : complex
        The radius of the semicircle. (Complex type for coercion
        performance.)
    w : complex
        The center of the semicircle.
    theta : complex
        The starting argument /angle of the semicircle. Usually `0` or
        `pi`.  (Complex type for coercion performance.)
    dtheta : complex
        The number of radians to travel where the sign of `dtheta`
        indicates direction. The absolute value of `dtheta` is equal to
        the arc length

    """
    def __init__(self, RiemannSurface RS, AnalyticContinuator AC,
                 complex x0, complex[:] y0, complex R, complex w,
                 complex theta, complex dtheta, int ncheckpoints=8):
        self.R = R
        self.w = w
        self.theta = theta
        self.dtheta = dtheta
        RiemannSurfacePathPrimitive.__init__(self, RS, AC, x0, y0,
                                             ncheckpoints=ncheckpoints)

    def __str__(self):
        return 'RiemannSurfacePathArc:\n' + \
            'radius:  %d\ncenter: %s\ntheta:  %d\ndtheta: %d'

    cpdef complex get_x(self, double t):
        return self.R*cexp(1.0j*(self.theta + t*self.dtheta)) + \
            self.w

    cpdef complex get_dxdt(self, double t):
        return (self.R*1.0j*self.dtheta) * \
            cexp(1.0j*(self.theta + t*self.dtheta))


cdef class RiemannSurfacePath(RiemannSurfacePathPrimitive):
    """A continuous, piecewise differentiable path on a Riemann surface.

    RiemannSurfacePath is a composite of
    :class:RiemannSurfacePathPrimitive objects. This path is
    parameterized for `t \in [0,1]`.
    """
    def __init__(self, RiemannSurface RS, complex x0, complex[:] y0,
                 RiemannSurfacePathPrimitive[:] segments):
        print '--- RSP __init__()'
        # RiemannSurfacePath delegates all analytic continuation to each
        # of its components, so we intialize its parent with a null
        # AnalyticContinuator object.
        #
        # Additionally, setting ncheckpoints to "0" prevents
        # self._initialize_checkpoints() from executing, which only
        # makes sense on a single path segment / path primitive.
        RiemannSurfacePathPrimitive.__init__(self, RS, None, x0, y0,
                                             ncheckpoints=0)

        # important: self.segments must be set after the parent
        # intialization call since parent will set self.segments equal
        # to "self"
        self.segments = segments
        self.nsegments = len(segments)
        print '--- RSP __init__() END'

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex get_x(self, double t):
        """Return the x-part of the path at `t \in [0,1]`.

        .. note::

           This RiemannSurfacePath is parameterized for `t \in
           [0,1]`. However, internally, each segment is separately
           parameterized for `t \in [0,1]`. This routine performs an
           appropriate scaling.
        """
        cdef RiemannSurfacePathPrimitive seg_k
        cdef complex x
        cdef int k

        if t == 1.0:
            k = self.nsegments-1
            t_seg = 1.0
        else:
            k = floor(t*self.nsegments)
            t_seg = t*self.nsegments - k

        seg_k = self.segments[k]
        x = seg_k.get_x(t_seg)
        return x

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex get_dxdt(self, double t):
        """Return the derivative of the x-part of the path at `t \in [0,1]`.

        .. note::

           This RiemannSurfacePath is parameterized for `t \in
           [0,1]`. However, internally, each segment is separately
           parameterized for `t \in [0,1]`. This routine performs an
           appropriate scaling.

        .. warning::

           Riemann surface paths are only piecewise differentiable and
           therefore may have discontinuous derivatives at the
           boundaries. Therefore, it may be more useful to perform
           segment-wise operations instead of operations on the
           whole of this object.

        """
        cdef RiemannSurfacePathPrimitive seg_k
        cdef complex dxdt
        cdef int k = floor(t*self.nsegments)

        t_seg = t*self.nsegments - k
        seg_k = self.segments[k]
        dxdt = seg_k.get_dxdt(t_seg)
        return dxdt

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex[:] get_y(self, double t):
        """Return the y-fibre of the path at `t \in [0,1]`.

        .. note::

           This RiemannSurfacePath is parameterized for `t \in
           [0,1]`. However, internally, each segment is separately
           parameterized for `t \in [0,1]`. This routine performs an
           appropriate scaling.
        """
        cdef RiemannSurfacePathPrimitive seg_k
        cdef complex[:] y
        cdef int k = floor(t*self.nsegments)

        t_seg = t*self.nsegments - k
        seg_k = self.segments[k]
        y = seg_k.get_y(t_seg)
        return y

