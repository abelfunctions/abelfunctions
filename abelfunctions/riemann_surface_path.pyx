#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

r"""Riemann Surface Paths :mod:`abelfunctions.riemann_surface_path`

Module for defining paths on Riemann surfaces. A basic Riemann surface path
consists of the Riemann surface on which it's defined, a
:class:`ComplexPathPrimitive` defining the x-projection of the path, and a
starting y-fibre. The first element of the y-fibre defines the starting point /
place of the surface. However, an entire ordered fibre of y-roots is requested
since many anlgorithms for analytic continuation require all roots.

Riemann surface paths are distinguished by how one analytically continues along
the path. Typically, if the complex path stays away from any discriminant
points of the Riemann surface then :class:`RiemannSurfaceSmale`, which is based
in Newton's method for root finding, can be used.

Classes
-------
.. autosummary::

  RiemannSurfacePathPrimitive
  RiemannSurfacePath
  RiemannSurfacePathPuiseux
  RiemannSurfacePathSmale

Functions
---------
.. autosummary::

  ordered_puiseux_series
  newton
  smale_alpha
  smale_beta
  smale_gamma

Contents
--------
"""

import warnings

import numpy
cimport numpy
cimport cython

from abelfunctions.divisor import DiscriminantPlace
from abelfunctions.puiseux import puiseux
from abelfunctions.utilities import matching_permutation
from abelfunctions import ComplexField as CDF

import mpmath

from sage.all import (
    QQ, QQbar, infinity, fast_callable, cached_method, cached_function)
from sage.symbolic.constants import pi as PI
from sage.functions.other import real_part, imag_part
from sage.plot.line import line

from sage.ext.interpreters.wrapper_el cimport Wrapper_el

cdef class RiemannSurfacePathPrimitive:
    r"""Primitive Riemann surface path object.

    Defines basic, primitive functionality for Riemann surface paths. Each path
    primitive is parameterized from :math:`t=0` to :math:`t=1`.

    Attributes
    ----------
    RS : RiemannSurface
        The Riemann surface on which this path primitive is defined.
    x0 : complex
        Starting x-value of the path.
    y0 : complex[]
        Starting y-fibre of the path.
    segments : list of RiemannSurfacePathPrimitive
        A list of the constituent components of the Riemann surface path.

    Methods
    -------
    .. autosummary::

      get_x
      get_dxds
      get_y
      plot_x
      plot_y
      plot3d_x
      plot3d_y

    """
    @property
    def riemann_surface(self):
        return self._riemann_surface

    @property
    def segments(self):
        return [self]

    @property
    def complex_path(self):
        return self._complex_path

    @property
    def x0(self):
        return self._x0

    @property
    def y0(self):
        return numpy.asarray(self._y0)

    def __init__(self, riemann_surface, complex_path, y0, ncheckpoints=16):
        r"""Initialize a Riemann surface path.

        This is a base class for the other classes in this module and should
        not be instantiated directly.

        Parameters
        ----------
        riemann_surface : RiemannSurface
            The Riemann surface on which the path is defined.
        complex_path : ComplexPathPrimitive
            The x-projection of the path.
        y0 : list of complex
            The starting fibre lying above the starting point of
            `complex_path`. The first component of the list indicates the
            starting sheet.
        ncheckpoints : int
            The number of points to cache analytic continuation results along
            the path so that one doesn't have to analytically continue from the
            start of the path every time.

        """
        self._segments = None
        self._nsegments = 1

        self._riemann_surface = riemann_surface
        self._complex_path = complex_path
        self._x0 = complex_path.eval(0)
        self._y0 = numpy.array(y0, dtype=object)

        # cached s, x, and y checkpoints
        self._ncheckpoints = ncheckpoints
        self._scheckpoints = numpy.zeros(ncheckpoints, dtype=object)
        self._xcheckpoints = numpy.zeros(ncheckpoints, dtype=object)
        self._ycheckpoints = numpy.zeros(
            (ncheckpoints, riemann_surface.deg), dtype=object)

        # initialize the checkpoints on the path. see
        # RiemannSurfacePath.__init__ for additional information on how the
        # following is interpreted in the composite setting.
        if ncheckpoints > 0:
            self._initialize_checkpoints()
        self._repr = None

    def __repr__(self):
        if not self._repr:
            self._set_repr()
        return self._repr

    def _set_repr(self):
        r"""Set the string representation of the Riemann surface path.

        This can only be done after object instantiation when we know the type
        of path created. String representation depends on whether self is a
        cycle on the surface or not.
        """
        is_cycle = False
        startx = self.x0
        endx = self.get_x(1.0)

        # cheap check: if the x-endpoints match
        if numpy.abs(startx - endx) < 1e-12:
            starty = self.y0
            endy = self.get_y(1.0)

            # expensive check: if the y-endpoints match
            if numpy.linalg.norm(starty - endy) < 1e-12:
                is_cycle = True

        if is_cycle:
            self._repr = 'Cycle on the %s'%(self.riemann_surface)
        else:
            self._repr = 'Path on the %s with x-projection %s'%(
                self.riemann_surface, self.complex_path)


    def __add__(self, other):
        r"""Add two Riemann surface paths together.

        Checks if the ending place of `self` is equal to the ending place of
        `other`. If so, returns a `RiemannSurfacePath` object whose segments
        are a concatenation of the path segments of each summand.

        Parameters
        ----------
        other : RiemannSurfacePathPrimitive

        Returns
        -------
        RiemannSurfacePath
        """
        # try getting the segments of the other object. Doing so asserts that
        # the other object is of type RiemannSurfacePathPrimitive
        try:
            segments = self.segments + other.segments
        except AttributeError:
            raise TypeError('Summands must both be of '
                            'RiemannSurfacePathPrimitive type.')

        # assert that the endpoint of the segments of this path matches with
        # those of the other path
        eps = 1e-8
        deg = self.riemann_surface.degree

        # get the ending place of the left RSPath (self) and the starting place
        # of the right RSPath (other).
        nsegments = len(self.segments)
        end_segment = self.segments[nsegments-1]
        x_end = end_segment.get_x(1.0)
        y_end = end_segment.get_y(1.0)
        x_start = other.x0
        y_start = other.y0

        # if the x- or y-values don't match, raise an error
        x_error = numpy.abs(x_start - x_end)
        y_error = numpy.linalg.norm(y_start - y_end)
        if (x_error > eps) or (y_error > eps):
            raise ValueError('Cannot form sum of paths: starting place and '
                             'fibre of right Riemann surface path does not '
                             'match ending place of left path.')

        complex_path = self.complex_path + other.complex_path
        gamma = RiemannSurfacePath(self.riemann_surface, complex_path,
                                   self.y0, segments)
        return gamma

    cpdef int _nearest_checkpoint_index(self, object s):
        r"""Returns the index of the checkpoint closest to and preceding `s`.

        Parameters
        ----------
        s : object
            Path parameter in the interval [0,1].

        Returns
        -------
        index : int
            The index `k` such that `self._scheckpoints[k] <= t` but
            `self._scheckpoints[k+1] > t`.
        """
        cdef object si
        cdef int index, k
        cdef int n = self._ncheckpoints
        if s == 1.0:
            return n-1
        for k in range(1,n):
            si = self._scheckpoints[k]
            if si >= s:
                index = k-1
                return index

        # use first checkpoint if something goes wrong
        index = 0
        return index

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
        cdef int n
        cdef object[:] s
        cdef object[:] x
        cdef object[:,:] y

        cdef object sim1, si
        cdef object xim1, xi
        cdef object[:] yim1, yi

        # initialize containers
        n = self._ncheckpoints
        s = numpy.linspace(0, 1, n, dtype=object)
        x = numpy.array([self.get_x(si) for si in s], dtype=object)
        y = numpy.zeros((n, self.riemann_surface.degree), dtype=object)

        # for each t-checkpoint compute the corresponding x- and y-checkpoint
        # by analytically continuing. note that the analytic continuation is
        # defined by the subclass and is not implemented in the base class
        sim1 = 0.0
        xim1 = self.x0
        yim1 = self.y0
        y[0,:] = yim1
        for i in range(1,n):
            si = s[i]
            xi = self._complex_path.eval(si)
            yi = self.analytically_continue(xim1, yim1, xi)
            y[i,:] = yi
            xim1 = xi
            yim1 = yi

        # store the checkpoint information
        self._scheckpoints = s
        self._xcheckpoints = x
        self._ycheckpoints = y

    cpdef object get_x(self, object s):
        r"""Return the x-part of the path at :math:`s \in [0,1]`.

        Parameters
        ----------
        s : object
            Path parameter in the interval [0,1].

        Returns
        -------
        value : complex
            The x-projection of self at s.

        """
        cdef object value = self._complex_path.eval(s)
        return value

    cpdef object get_dxds(self, object s):
        r"""Return the derivative of the x-part of the path at :math:`s \in [0,1]`.

        Parameters
        ----------
        s : object
            Path parameter in the interval [0,1].

        Returns
        -------
        value : complex
            The derivative of the x-projection of self at s.

        """
        cdef value = self._complex_path.derivative(s)
        return value

    cpdef object[:] get_y(self, object s):
        r"""Return the y-fibre of the path at :math:`s \in [0,1]`.

        Delegates to :meth:`analytically_continue`.

        Parameters
        ----------
        s : object
            Path parameter in the interval [0,1].

        Returns
        -------
        value : complex[:]
            The y-fibre above the path at s.
        """
        cdef int i
        cdef object xim1, xi
        cdef object[:] yim1, yi

        # get the closest checkpoint to the desired t-value
        i = self._nearest_checkpoint_index(s)
        xim1 = self._xcheckpoints[i]
        yim1 = self._ycheckpoints[i]

        # analytically continue to target
        xi = self._complex_path.eval(s)
        yi = self.analytically_continue(xim1, yim1, xi)
        return yi

    @cython.boundscheck(False)
    cpdef object[:] analytically_continue(self, object xi, object[:] yi, object xip1):
        raise NotImplementedError('Implement in subclass.')

    def integrate(self, omega):
        r"""Integrate `omega` along this path.

        Parameters
        ----------
        omega : Differential
            A differential (one-form) on the Riemann surface.

        Returns
        -------
        integral : complex
           The integral of omega along self.
        """
        omega_gamma = self.parameterize(omega)
        mpmath.mp.prec = CDF().prec()  # Update the precision before integrating
        integral = mpmath.quad(omega_gamma, [0, 1])
        return CDF(integral.real) + CDF(1j) * CDF(integral.imag)

    def evaluate(self, omega, s):
        r"""Evaluates `omega` along the path at :math:`s \in [0,1]`.

        Parameters
        ----------
        omega : Differential
            A differential (one-form) on the Riemann surface.
        s : object or object[:]
            Path parameter(s) in the interval [0,1].

        Returns
        -------
        values : complex or complex[:]
            The differential omega evaluated along the path at each of the
            points in `s`.
        """
        omega_gamma = self.parameterize(omega)
        values = omega_gamma(s)
        return values

    def parameterize(self, omega):
        raise NotImplementedError('Implement in subclass.')

    def plot_x(self, *args, **kwds):
        r"""Plot the x-part of the path in the complex x-plane.

        Calls :func:`ComplexPath.plot` on this path's x-projection.

        Parameters
        ----------
        *args : list
            Arguments passed to :func:`ComplexPath.plot`.
        **kwds : dict
            Keywords passed to :func:`ComplexPath.plot`.
        Returns
        -------
        matplotlib lines array.

        """
        return self.complex_path.plot(*args, **kwds)

    def plot_y(self, plot_points=128, **kwds):
        r"""Plot the y-part of the path in the complex y-plane.

        Additional arguments and keywords are passed to
        ``matplotlib.pyplot.plot``.

        Parameters
        ----------
        N : int
            The number of interpolating points used to plot.
        t0 : object
            Starting t-value in [0,1].
        t1 : object
            Ending t-value in [0,1].

        Returns
        -------
        plt : Sage plot.
            A plot of the complex y-projection of the path.

        """
        s = numpy.linspace(0, 1, plot_points, dtype=object)
        vals = numpy.array([self.get_y(si)[0] for si in s], dtype=object)
        pts = [(real_part(y), imag_part(y)) for y in vals]
        plt = line(pts, **kwds)
        return plt


cdef class RiemannSurfacePath(RiemannSurfacePathPrimitive):
    r"""A composite of Riemann surface path primitives.

    These are usually created via summation of other paths, such as
    :class:`RiemannSurfacePathPuiseux` and :class:`RiemannSurfacePathSmale`,
    and represent a composite of :class:`RiemannSurfacePathPrimitives`.

    Attributes
    ----------
    segments : list of RiemannSurfacePathPrimitives
        A list of the constituent paths that make up this composite path.

    Methods
    -------
    .. autosummary::

      segment_index_at_parameter

    """

    @property
    def segments(self):
        return numpy.asarray(self._segments).tolist()

    def __init__(self, riemann_surface, complex_path, y0, segments):
        r"""Directly instantiate a RiemannSurfacePath from a Riemann surface and a list
        of Riemann surface path primitives.

        Parameters
        ----------
        riemann_surface : RiemannSurface
            The Riemann surface on which the path is defined.
        complex_path : ComplexPathPrimitive
            The x-projection of the path.
        y0 : list of complex
            The starting fibre lying above the starting point of
            `complex_path`. The first component of the list indicates the
            starting sheet.
        *args : list
            A list of :class:`RiemannSurfacePathPrimitive`s which make up the
            segments / constituents of this path.
        """
        # RiemannSurfacePath delegates all analytic continuation to each of its
        # components.
        #
        # Additionally, setting ncheckpoints to "0" prevents
        # self._initialize_checkpoints() from executing, which only makes sense
        # on a single path segment / path primitive.
        RiemannSurfacePathPrimitive.__init__(
            self, riemann_surface, complex_path, y0, ncheckpoints=0)

        self._segments = numpy.array(segments, dtype=RiemannSurfacePathPrimitive)
        self._nsegments = len(segments)

    def __getitem__(self, int index):
        return self._segments[index]

    cdef int segment_index_at_parameter(self, object s):
        r"""Returns the index of the complex path segment located at the given
        parameter :math:`s \in [0,1]`.

        Parameters
        ----------
        s : float
            Path parameter in the interval [0,1].

        Returns
        -------
        index : int
            The index `k` of the path segment :math:`\gamma_k`.
        """
        # the following is a fast way to divide the interval [0,1] into n
        # partitions and determine which partition s lies in. since this is
        # done often it needs to be fast
        cdef int k = numpy.floor(s*self._nsegments)
        cdef int diff = (self._nsegments - 1) - k
        cdef int dsgn = diff >> 31
        cdef int index = k + (diff & dsgn)
        return index

    cpdef object get_x(self, object s):
        r"""Return the x-part of the path at :math:`s \in [0,1]`.

        Parameters
        ----------
        s : object
            Path parameter in the interval [0,1].

        Returns
        -------
        value : complex
            The x-projection of self at s.

        """
        cdef int k = self.segment_index_at_parameter(s)
        cdef object s_segment = s*self._nsegments - k
        cdef RiemannSurfacePathPrimitive segment = self._segments[k]
        cdef object value = segment.get_x(s_segment)
        return value

    cpdef object get_dxds(self, object s):
        r"""Return the derivative of the x-part of the path at :math:`s \in [0,1]`.

        Parameters
        ----------
        s : object
            Path parameter in the interval [0,1].

        Returns
        -------
        value : complex
            The derivative of the x-projection of self at s.

        """
        cdef int k = self.segment_index_at_parameter(s)
        cdef object s_segment = s*self._nsegments - k
        cdef RiemannSurfacePathPrimitive segment = self._segments[k]
        cdef object value = segment.get_dxds(s_segment)
        return value

    cpdef object[:] get_y(self, object s):
        r"""Return the y-fibre of the path at :math:`s \in [0,1]`.

        Delegates to :meth:`analytically_continue`.

        Parameters
        ----------
        s : object
            Path parameter in the interval [0,1].

        Returns
        -------
        value : complex[:]
            The y-fibre above the path at s.
        """
        cdef int k = self.segment_index_at_parameter(s)
        cdef object s_segment = s*self._nsegments - k
        cdef RiemannSurfacePathPrimitive segment = self._segments[k]
        cdef object[:] value = segment.get_y(s_segment)
        return value

    def parameterize(self, omega):
        r"""Returns the differential omega parameterized on the path.

        Given a differential math:`\omega = \omega(x,y)dx`, `parameterize`
        returns the differential

        .. math::

            \omega_\gamma(s) = \omega(\gamma_x(s),\gamma_y(s)) \gamma_x'(s)

        where :math:`s \in [0,1]` and :math:`\gamma_x,\gamma_y` and the x- and
        y-components of the path `\gamma` using this analytic continuator.

        .. note::

            This may be pretty slow in the composite path case. Usually, we
            want to integrate using parameterizations which, in the composite
            case, we just perform one segment at a time instead of determining
            a global parameterization here.

        Parameters
        ----------
        omega : Differential
            A differential (one-form) on the Riemann surface.

        Returns
        -------
        omega_gamma : function
            The differential parameterized on the curve for s in the interval
            [0,1].
        """
        def omega_gamma(s):
            cdef int k
            k = self.segment_index_at_parameter(s)
            s_segment = s*self._nsegments - k
            segment = self._segments[k]
            omega_gamma_segment = segment.parameterize(omega)
            return omega_gamma_segment(s_segment)
        return omega_gamma

    def evaluate(self, omega, s):
        r"""Evaluates `omega` along the path at :math:`s \in [0,1]`.

        Parameters
        ----------
        omega : Differential
            A differential (one-form) on the Riemann surface.
        s : object[:]
            Path parameter(s) in the interval [0,1].

        Returns
        -------
        values : complex or complex[:]
            The differential omega evaluated along the path at each of the
            points in `s`.
        """
        # determine the number of points per segment on which to evaluate
        cdef int k
        N = len(s)
        nsegs = len(self._segments)
        ppseg = int(N/nsegs)
        values = numpy.zeros(nsegs*ppseg, dtype=object)
        values_seg = numpy.zeros(ppseg, dtype=object)
        tseg = numpy.linspace(0, 1, ppseg)

        # evaluate along each segment
        for k in range(nsegs):
            segment = self._segments[k]
            values_seg = segment.evaluate(omega, tseg)
            for j in range(ppseg):
                values[k*ppseg+j] = values_seg[j]
        return values

    def integrate(self, omega):
        r"""Integrate `omega` along this path.

        Parameters
        ----------
        omega : Differential
            A differential (one-form) on the Riemann surface.

        Returns
        -------
        integral : complex
            The integral of omega along self.
        """
        integral = CDF(0.0)
        for gamma in self._segments:
            integral += gamma.integrate(omega)
        return integral


##############################################
# Puiseux Series-based Riemann Surface Paths #
##############################################
def ordered_puiseux_series(riemann_surface, complex_path, y0, target_point):
    r"""Returns an ordered list of Puiseux series such that each Puiseux series
    matches with the corresponding y-fibre element above the starting point of
    `complex_path`.

    In order to analytically continue from the regular places at the beginning
    of the path :math:`x=a` to the discriminant places at the end of the path
    :math:`x=b`we need to compute all of the `PuiseuxXSeries` at :math:`x=b`.
    There are two steps to this calculation:

    * compute enough terms of the Puiseux series centered at :math:`x=b` in
      order to accurately capture the y-roots at :math:`x=a`.

    * permute the series accordingly to match up with the y-roots at
      :math:`x=a`.

    Parameters
    ----------
    riemann_surface : RiemannSurface
        The riemann surface on which all of this lives.
    complex_path : ComplexPath
        The path or path segment starting at a regular point and ending at a
        discriminant point.
    y0 : list of complex
        The starting fibre lying above the starting point of `complex_path`.
        The first component of the list indicates the starting sheet.
    target_point : complex
        The point to analytically continue to. Usually a discriminant point.

    Methods
    -------
    .. autosummary::

      analytically_continue

    Returns
    -------
    list, Place : a list of Puiseux series and a Place
        A list of ordered Puiseux series corresponding to each branch above
        :math:`x=a` as well as the place that the first y-fibre element
        analytically continues to.
    """
    # obtain all puiseux series above the target place
    f = riemann_surface.f
    x0 = CDF(complex_path.eval(0)) # XXX - need to coerce input to CC
    y0 = numpy.array(y0, dtype=object)
    P = puiseux(f, target_point)

    # extend the Puiseux series to enough terms to accurately captue the
    # y-fibre above x=a (the starting point of the complex path)
    for Pi in P:
        Pi.extend_to_x(x0)

    # compute the corresponding x-series representations of the Puiseux series
    alpha = 0 if target_point == infinity else target_point
    px = [Pi.xseries() for Pi in P]
    p = [pxi for sublist in px for pxi in sublist]
    ramification_indices = [Pi.ramification_index for Pi in P]

    # reorder them according to the ordering of the y-fibre above x=x0
    p_evals_above_x0 = [pj(x0-alpha) for pj in p]
    p_evals_above_x0 = numpy.array(p_evals_above_x0, dtype=object)
    sigma = matching_permutation(p_evals_above_x0, y0)
    p = sigma.action(p)

    # also return the place that the first y-fibre element ends up analytically
    # continuing to
    px_idx = sigma[0] # index of target x-series in unsorted list
    place_idx = -1    # index of place corresponding to this x-series
    while px_idx >= 0:
        place_idx += 1
        px_idx -= abs(ramification_indices[place_idx])
    target_place = DiscriminantPlace(riemann_surface,P[place_idx])
    return p, target_place


cdef class RiemannSurfacePathPuiseux(RiemannSurfacePathPrimitive):
    r"""A Riemann surface path that uses Puiseux series to analytically continue
    along a complex path.

    Newton's method / Smale's alpha theory (see
    :class:`RiemannSurfacePathSmale`) breaks down when close to a discriminant
    point of the curve since the y-sheets coalesce at that point. In order to
    accurately track the y-fibre above points near the discrimimant point we
    need to compute an ordered set of puiseux series

    Attributes
    ----------
    puiseux_series : list of PuiseuxSeries
        An ordered list of Puiseux series centered at the endpoint of the complex path.
    center : complex
        The center of the above Puiseux series. (The endpoint of the complex path.)
    target_point : complex
        The point in the complex x-plane that we're analytically coninuing to.
    target_place : Place
        The place that the first sheet of the input y-fibre ends up
        analytically continuing to.

    Methods
    -------
    .. autosummary::

      analytically_continue
      parameterize

    See Also
    --------
    * :func:`ordered_puiseux_series`
    * :class:`RiemannSurfacePathSmale`

    """
    def __init__(self, riemann_surface, complex_path, y0, ncheckpoints=16):
        # if the complex path leads to a discriminant point then get the exact
        # representation of said discrimimant point
        target_point = complex_path.eval(1)
        if target_point in [numpy.Infinity, infinity]:
            target_point = infinity
        elif abs(CDF(target_point)) > 1e12:
            target_point = infinity
        else:
            discriminant_point = riemann_surface.path_factory.closest_discriminant_point(target_point)
            if abs(CDF(target_point - discriminant_point)) < 1e-12:
                target_point = discriminant_point
            else:
                # if it's not discriminant then try to coerce to QQ or QQbar
                try:
                    target_point = QQ(target_point)
                except TypeError:
                    try:
                        target_point = QQbar(target_point)
                    except TypeError:
                        pass

        # compute and store the ordered puiseux series needed to analytically
        # continue as well as the target place for parameterization purposes
        puiseux_series, target_place = ordered_puiseux_series(
            riemann_surface, complex_path, y0, target_point)
        self.puiseux_series = puiseux_series
        self.target_point = target_point
        self.target_place = target_place

        # now that the machinery is set up we can instantiate the base object
        RiemannSurfacePathPrimitive.__init__(
            self, riemann_surface, complex_path, y0, ncheckpoints=ncheckpoints)

    @cython.boundscheck(False)
    cpdef object[:] analytically_continue(self, object xi, object[:] yi, object xip1):
        r"""Analytically continue the y-fibre `yi` lying above `xi` to the y-fibre
        lying above `xip1`.

        We analytically continue by simply evaluating the ordered puiseux
        series computed during initialization of the Riemann surface path.

        Parameters
        ----------
        xi : complex
            The starting complex x-value.
        yi: complex[:]
            The starting complex y-fibre lying above `xi`.
        xip1: complex
            The target complex x-value.

        Returns
        -------
        yi : complex[:]
            The corresponding y-fibre lying above `xi`.
        """
        # XXX HACK - need to coerce input to CC for puiseux series to evaluate
        xi = CDF(xi)
        xip1 = CDF(xip1)

        # return the current fibre if the step size is too small
        if numpy.abs(xip1-xi) < 1e-15:
            return yi

        # simply evaluate the ordered puiseux series at xip1
        alpha = CDF(0) if self.target_point == infinity else CDF(self.target_point)
        cdef object[:] yip1 = numpy.array(
            [pj(xip1-alpha) for pj in self.puiseux_series],
            dtype=object)
        return yip1

    @cached_method
    def parameterize(self, omega):
        r"""Returns the differential omega parameterized on the path.

        Given a differential math:`\omega = \omega(x,y)dx`, `parameterize`
        returns the differential

        .. math::

            \omega_\gamma(s) = \omega(\gamma_x(s),\gamma_y(s)) \gamma_x'(s)

        where :math:`s \in [0,1]` and :math:`\gamma_x,\gamma_y` and the x- and
        y-components of the path `\gamma` using this analytic continuator.

        Parameters
        ----------
        omega : Differential
            A differential (one-form) on the Riemann surface.

        Returns
        -------
        omega_gamma : function
            The differential parameterized on the curve for s in the interval
            [0,1].
        """
        # localize the differential at the discriminant place
        P = self.target_place
        omega_local = omega.localize(P)
        omega_local = omega_local.laurent_polynomial().change_ring(CDF())

        # extract relevant information about the Puiseux series
        p = P.puiseux_series
        x0 = (self._x0)
        y0 = (self._y0[0])
        alpha = 0 if self.target_point == infinity else self.target_point
        xcoefficient = (p.xcoefficient)
        e = numpy.int(p.ramification_index)

        # the parameter of the path s \in [0,1] does not necessarily match with
        # the local coordinate t of the place. perform the appropriate scaling
        # on the integral.
        tprim = ((x0-alpha)/xcoefficient)**(CDF(1.).real()/e)
        unity = [numpy.exp(CDF(2.j)*PI*k/abs(e)) for k in range(abs(e))]
        tall = [unity[k]*tprim for k in range(abs(e))]
        ytprim = numpy.array([p.eval_y(tk) for tk in tall], dtype=numpy.object)
        k = numpy.argmin(numpy.abs(ytprim - y0))
        tcoefficient = tall[k]

        # XXX HACK - CC coercion
        tcoefficient = CDF(tcoefficient)
        def omega_gamma(s):
            s = CDF(s)
            dtds = -tcoefficient
            val = omega_local(tcoefficient*(CDF(1-s))) * dtds
            return (val)
        return numpy.vectorize(omega_gamma, otypes=[object])


####################################################
# Smale's Alpha Theory-based Riemann Surface Paths #
####################################################
###cpdef double ABELFUNCTIONS_SMALE_ALPHA0 = 1.1884471871911697 # = (13-2*sqrt(17))/4

cdef int factorial(int n):
    cdef int result = 1
    cdef int i
    for i in range(1,n+1):
        result *= i
    return result

def newton(df, xip1, yij):
    """Newton iterate a y-root yij of a polynomial :math:`f = f(x,y)`, lying above
    some x-point xi, to the x-point xip1.

    Given :math:`f(x_i,y_{i,j}) = 0` and some complex number :math:`x_{i+1}`,
    this function returns a complex number :math:`y_{i+1,j}` such that
    :math:`f(x_{i+1},y_{i+1,j}) = 0`.

    Parameters
    ----------
    df : list of polynomials
        A list of all of the y-derivatives of f, including f itself.
    xip1 : complex
        The x-point to analytically continue to.
    yij: complex
        A y-root at xi. The root that we'll analytically continue.

    Returns
    -------
    yij : complex
        A y-root of f lying above `xip1`.

    """
    df0 = df[0]
    df1 = df[1]
    step = CDF(1.0)
    maxIter = 1000
    numIter = 0
    while numpy.abs(step) > 1e-14:
        # if df is not invertible then we are at a critical point.
        df1y = df1(xip1,yij)
        if numpy.abs(df1y) < 1e-14:
            break
        step = df0(xip1,yij) / df1y
        yij = yij - step

        numIter += 1
        if numIter >= maxIter:
            warnings.warn('Newton failed to converge after %d iterations. Final step size: %g' % (maxIter, numpy.abs(step)))
            break
    return yij


cdef object smale_beta(Wrapper_el[:] df, object xip1, object yij):
    """Smale beta function.

    The Smale beta function is simply the size of a Newton iteration

    Parameters
    ---------
    df : list of polynomials
        A list of all of the y-derivatives of f, including f itself.
    xip1 : complex
        The x-point to analytically continue to.
    yij: complex
        A y-root at xi. The root that we'll analytically continue.

    Returns
    -------
    val : object
        :math:`\beta(f,x_{i+1},y_{i,j})`.
    """
    cdef Wrapper_el df0 = df[0]
    cdef Wrapper_el df1 = df[1]
    cdef object val
    cdef object numer, denom

    numer = df0(xip1, yij)
    denom = df1(xip1, yij)
    val = numpy.abs(numer / denom)
    return val


cdef object smale_gamma(Wrapper_el[:] df, object xip1, object yij, int degree):
    """Smale gamma function.

    Parameters
    ----------
    df : MultivariatePolynomial
        A list of all of the y-derivatives of f, including f itself.
    xip1 : complex
        The x-point to analytically continue to.
    yij : complex
        A y-root at xi. The root that we'll analytically continue.
    degree : int
        The degree of f.

    Returns
    -------
    object
        The Smale gamma function.
    """
    cdef Wrapper_el df0 = df[0]
    cdef Wrapper_el df1 = df[1]
    cdef Wrapper_el dfn
#    cdef int deg = len(df) - 1
    cdef object df1y
    cdef object gamma, gamman, numer
    cdef int n

    gamma = 0
    df1y = df1(xip1, yij)
    for n in range(2,degree+1):
        dfn = df[n]
        numer = numpy.abs(dfn(xip1, yij))
        gamman = numer / (factorial(n)*numpy.abs(df1y))
        gamman = gamman**(CDF(1.0).real()/(n-CDF(1.0).real()))
        if gamman > gamma:
            gamma = gamman
    return gamma


cdef object smale_alpha(Wrapper_el[:] df, object xip1, object yij, int degree):
    """Smale alpha function.

    Parameters
    ----------
    df : MultivariatePolynomial
        a list of all of the y-derivatives of f (up to the y-degree)
    xip1 : complex
        the x-point to analytically continue to
    yij : complex
        a y-root at xi. The root that we'll analytically continue.

    Returns
    -------
    object
        The Smale alpha function.
    """
    cdef object val = smale_beta(df,xip1,yij) * smale_gamma(df,xip1,yij,degree)
    return val



cdef class RiemannSurfacePathSmale(RiemannSurfacePathPrimitive):
    r"""A Riemann surface Path that uses Smale's alpha theory with Newton iteration
    to analytically continue along a complex path.

    For complex paths that stay sufficiently away from a discrimimant point of
    the curve defining the Riemann surface we can use Newton's method to
    analytically continue. This method is fast compared to calculation and
    evaluation of Puiseux series.

    Smale's alpha theory is used to ensure that appropriate steps sizes are
    taken in the complex x-plane. (i.e. along the complex path.)

    Attributes
    ----------
    degree : int
        The y-degree of the underlying curve of the Riemann surface.
    df : list of functions
        A list of all of the y-derivatives of f *up to the y-degree).

    Methods
    -------
    .. autosummary::

      analytically_continue
      parameterize

    """
    @property
    def degree(self):
        return self._degree

    @property
    def df(self):
        return self._df

    def __init__(self, riemann_surface, complex_path, y0, ncheckpoints=32):
        # store a list of all y-derivatives of f (including the zeroth deriv)
        #
        # it is very important that the domain of the fast_callable versions of
        # f and its derivatives is CDF.
        degree = riemann_surface.degree
        f = riemann_surface.f.change_ring(CDF())
        x,y = f.parent().gens()
        df = [
            fast_callable(f.derivative(y,k), vars=(x,y), domain=CDF())
            for k in range(degree+1)
        ]
        df = numpy.array(df, dtype=object)

        self._degree = degree
        self._df = df
        RiemannSurfacePathPrimitive.__init__(
            self, riemann_surface, complex_path, y0, ncheckpoints=ncheckpoints)

    @cython.boundscheck(False)
    cpdef object[:] analytically_continue(self, object xi, object[:] yi, object xip1):
        cdef int j,k
        cdef object betaij, betaik, distancejk, alpha
        cdef object xiphalf, yij, yik
        cdef object[:] yiphalf, yip1

        # return the current fibre if the step size is too small
        if numpy.abs(xip1-xi) < 1e-14:
            return yi

        # first determine if the y-fibre guesses are 'approximate solutions'.
        # if any of them are not then refine the step by analytically
        # continuing to an intermediate "time"
        for j in range(self._degree):
            yij = yi[j]
            alpha = smale_alpha(self._df, xip1, yij, self._degree)
            if alpha > (13-2*(CDF(17)**CDF(0.5)))/4:
                xiphalf = (xi + xip1)/CDF(2.0).real()
                yiphalf = self.analytically_continue(xi, yi, xiphalf)
                yip1 = self.analytically_continue(xiphalf, yiphalf, xip1)
                return yip1

        # next, determine if the approximate solutions will converge to
        # different associated solutions
        for j in range(self._degree):
            yij = yi[j]
            betaij = smale_beta(self._df, xip1, yij)
            for k in range(j+1, self._degree):
                yik = yi[k]
                betaik = smale_beta(self._df, xip1, yik)
                distancejk = numpy.abs(yij-yik)
                if distancejk < 2.5*(betaij + betaik):  # 2*beta
                    # approximate solutions don't lead to distinct roots.
                    # refine the step by analytically continuing to an
                    # intermedite time
                    xiphalf = (xi + xip1)/CDF(2.0).real()
                    yiphalf = self.analytically_continue(xi, yi, xiphalf)
                    yip1 = self.analytically_continue(xiphalf, yiphalf, xip1)
                    return yip1

        # finally, since we know that we have approximate solutions that will
        # converge to difference associated solutions we will Netwon iterate
        yip1 = numpy.empty(self._degree, dtype=object)
        for j in range(self._degree):
            yip1[j] = newton(self._df, xip1, yi[j])
        return yip1

    @cached_method
    def parameterize(self, omega):
        r"""Returns the differential omega parameterized on the path.

        Given a differential math:`\omega = \omega(x,y)dx`, `parameterize`
        returns the differential

        .. math::

            \omega_\gamma(s) = \omega(\gamma_x(s),\gamma_y(s)) \gamma_x'(s)

        where :math:`s \in [0,1]` and :math:`\gamma_x,\gamma_y` and the x- and
        y-components of the path `\gamma` using this analytic continuator.

        Parameters
        ----------
        omega : Differential
            A differential (one-form) on the Riemann surface.

        Returns
        -------
        omega_gamma : function
            The differential parameterized on the curve for s in the interval
            [0,1].
        """
        def omega_gamma(s):
            xs = self.get_x(s)
            ys = self.get_y(s)[0]
            dxds = self.get_dxds(s)
            return omega(xs,ys) * dxds
        return numpy.vectorize(omega_gamma, otypes=[object])
