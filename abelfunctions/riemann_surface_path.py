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
import scipy
from abelfunctions.divisor import DiscriminantPlace
from abelfunctions.puiseux import puiseux
from abelfunctions.utilities import matching_permutation
from numpy import double, complex

from sage.all import (QQ, QQbar, CC, infinity, fast_callable,
                      factorial, cached_method)
from sage.functions.other import real_part, imag_part, floor
from sage.plot.line import line


class RiemannSurfacePathPrimitive(object):
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
    def segments(self):
        return [self]

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
        self.riemann_surface = riemann_surface
        self.complex_path = complex_path
        self.x0 = complex_path(0)
        self.y0 = numpy.array(y0, dtype=complex)

        # cached s, x, and y checkpoints
        self._ncheckpoints = ncheckpoints
        self._scheckpoints = numpy.zeros(ncheckpoints, dtype=double)
        self._xcheckpoints = numpy.zeros(ncheckpoints, dtype=complex)
        self._ycheckpoints = numpy.zeros(
            (ncheckpoints, riemann_surface.deg), dtype=complex)

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
        # the other object is of type RiemannSurfacePathPrimitve
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
        end_segment = self.segments[-1]
        x_end = end_segment.get_x(1.0)
        y_end = end_segment.get_y(1.0)
        x_start = other.x0
        y_start = other.y0

        # if the x- or y-values don't match, raise an error
        x_error = abs(x_start - x_end)
        y_error = numpy.linalg.norm(y_start - y_end)
        if (x_error > eps) or (y_error > eps):
            raise ValueError('Cannot form sum of paths: starting place and '
                             'fibre of right Riemann surface path does not '
                             'match ending place of left path.')

        complex_path = self.complex_path + other.complex_path
        gamma = RiemannSurfacePath(self.riemann_surface, complex_path,
                                   self.y0, segments)
        return gamma

    def _nearest_checkpoint_index(self, s):
        r"""Returns the index of the checkpoint closest to and preceding `s`.

        Parameters
        ----------
        s : double
            Path parameter in the interval [0,1].

        Returns
        -------
        index : int
            The index `k` such that `self._scheckpoints[k] <= t` but
            `self._scheckpoints[k+1] > t`.
        """
        n = self._ncheckpoints
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
        # initialize containers
        n = self._ncheckpoints
        s = numpy.linspace(0, 1, n, dtype=double)
        x = numpy.array([self.get_x(si) for si in s], dtype=complex)
        y = numpy.zeros((n, self.riemann_surface.degree), dtype=complex)

        # for each t-checkpoint compute the corresponding x- and y-checkpoint
        # by analytically continuing. note that the analytic continuation is
        # defined by the subclass and is not implemented in the base class
        sim1 = 0.0
        xim1 = self.x0
        yim1 = self.y0
        y[0,:] = yim1
        for i in range(1,n):
            si = s[i]
            xi = self.complex_path(si)
            yi = self.analytically_continue(xim1, yim1, xi)
            y[i,:] = yi
            xim1 = xi
            yim1 = yi

        # store the checkpoint information
        self._scheckpoints = s
        self._xcheckpoints = x
        self._ycheckpoints = y

    def get_x(self, s):
        r"""Return the x-part of the path at :math:`s \in [0,1]`.

        Parameters
        ----------
        s : double
            Path parameter in the interval [0,1].

        Returns
        -------
        value : complex
            The x-projection of self at s.

        """
        value = self.complex_path.eval(s)
        return value

    def get_dxds(self, s):
        r"""Return the derivative of the x-part of the path at :math:`s \in [0,1]`.

        Parameters
        ----------
        s : double
            Path parameter in the interval [0,1].

        Returns
        -------
        value : complex
            The derivative of the x-projection of self at s.

        """
        value = self.complex_path.derivative(s)
        return value

    def get_y(self, s):
        r"""Return the y-fibre of the path at :math:`s \in [0,1]`.

        Delegates to :meth:`analytically_continue`.

        Parameters
        ----------
        s : double
            Path parameter in the interval [0,1].

        Returns
        -------
        value : complex[:]
            The y-fibre above the path at s.
        """
        # get the closest checkpoint to the desired t-value
        i = self._nearest_checkpoint_index(s)
        xim1 = self._xcheckpoints[i]
        yim1 = self._ycheckpoints[i]

        # analytically continue to target
        xi = self.complex_path.eval(s)
        yi = self.analytically_continue(xim1, yim1, xi)
        return yi

    def analytically_continue(self, xi, yi, xip1):
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
        integral = scipy.integrate.romberg(omega_gamma, 0.0, 1.0)
        return integral

    def evaluate(self, omega, s):
        r"""Evaluates `omega` along the path at :math:`s \in [0,1]`.

        Parameters
        ----------
        omega : Differential
            A differential (one-form) on the Riemann surface.
        s : double or double[:]
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
        t0 : double
            Starting t-value in [0,1].
        t1 : double
            Ending t-value in [0,1].

        Returns
        -------
        plt : Sage plot.
            A plot of the complex y-projection of the path.

        """
        s = numpy.linspace(0, 1, plot_points, dtype=double)
        vals = numpy.array([self.get_y(si)[0] for si in s], dtype=complex)
        pts = [(real_part(y), imag_part(y)) for y in vals]
        plt = line(pts, **kwds)
        return plt



class RiemannSurfacePath(RiemannSurfacePathPrimitive):
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
        return self._segments

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

        self._segments = segments
        self._nsegments = len(segments)

    def __getitem__(self, index):
        return self._segments[index]

    def segment_index_at_parameter(self, s):
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
        k = floor(s*self._nsegments)
        diff = (self._nsegments - 1) - k
        dsgn = diff >> 31
        return k + (diff & dsgn)

    def get_x(self, s):
        r"""Return the x-part of the path at :math:`s \in [0,1]`.

        Parameters
        ----------
        s : double
            Path parameter in the interval [0,1].

        Returns
        -------
        value : complex
            The x-projection of self at s.

        """
        k = self.segment_index_at_parameter(s)
        s_segment = s*self._nsegments - k
        segment = self._segments[k]
        value = segment.get_x(s_segment)
        return value

    def get_dxds(self, s):
        r"""Return the derivative of the x-part of the path at :math:`s \in [0,1]`.

        Parameters
        ----------
        s : double
            Path parameter in the interval [0,1].

        Returns
        -------
        value : complex
            The derivative of the x-projection of self at s.

        """
        k = self.segment_index_at_parameter(s)
        s_segment = s*self._nsegments - k
        segment = self._segments[k]
        value = segment.get_dxds(s_segment)
        return value

    def get_y(self, s):
        r"""Return the y-fibre of the path at :math:`s \in [0,1]`.

        Delegates to :meth:`analytically_continue`.

        Parameters
        ----------
        s : double
            Path parameter in the interval [0,1].

        Returns
        -------
        value : complex[:]
            The y-fibre above the path at s.
        """
        k = self.segment_index_at_parameter(s)
        s_segment = s*self._nsegments - k
        segment = self._segments[k]
        value = segment.get_y(s_segment)
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
        s : double[:]
            Path parameter(s) in the interval [0,1].

        Returns
        -------
        values : complex or complex[:]
            The differential omega evaluated along the path at each of the
            points in `s`.
        """
        # determine the number of points per segment on which to evaluate
        N = len(s)
        nsegs = len(self._segments)
        ppseg = int(N/nsegs)
        values = numpy.zeros(nsegs*ppseg, dtype=complex)
        values_seg = numpy.zeros(ppseg, dtype=complex)
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
        integral = complex(0.0)
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
    x0 = CC(complex_path(0)) # XXX - need to coerce input to CC
    y0 = numpy.array(y0, dtype=complex)
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
    p_evals_above_x0 = numpy.array(p_evals_above_x0, dtype=complex)
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


class RiemannSurfacePathPuiseux(RiemannSurfacePathPrimitive):
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
        target_point = complex_path(1)
        if target_point in [numpy.Infinity, infinity]:
            target_point = infinity
        elif abs(CC(target_point)) > 1e12:
            target_point = infinity
        else:
            discriminant_point = riemann_surface.path_factory.closest_discriminant_point(target_point)
            if abs(CC(target_point - discriminant_point)) < 1e-12:
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

    def analytically_continue(self, xi, yi, xip1):
        r"""Analytically continue the y-fibre `yi` lying above `xi` to the y-fibre lying
        above `xip1`.

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
        xi = CC(xi)
        xip1 = CC(xip1)

        # return the current fibre if the step size is too small
        if numpy.abs(xip1-xi) < 1e-15:
            return yi

        # simply evaluate the ordered puiseux series at xip1
        alpha = CC(0) if self.target_point == infinity else CC(self.target_point)
        yip1 = [pj(xip1-alpha) for pj in self.puiseux_series]
        yip1 = numpy.array(yip1, dtype=complex)
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
        omega_local = omega_local.laurent_polynomial().change_ring(CC)

        # extract relevant information about the Puiseux series
        p = P.puiseux_series
        x0 = complex(self.gamma.x0)
        y0 = complex(self.gamma.y0[0])
        alpha = 0 if self.target_point == infinity else self.target_point
        xcoefficient = complex(p.xcoefficient)
        e = numpy.int(p.ramification_index)

        # the parameter of the path s \in [0,1] does not necessarily match with
        # the local coordinate t of the place. perform the appropriate scaling
        # on the integral.
        tprim = complex((x0-alpha)/xcoefficient)**(1./e)
        unity = [numpy.exp(2.j*numpy.pi*k/abs(e)) for k in range(abs(e))]
        tall = [unity[k]*tprim for k in range(abs(e))]
        ytprim = numpy.array([p.eval_y(tk) for tk in tall], dtype=numpy.complex)
        k = numpy.argmin(numpy.abs(ytprim - y0))
        tcoefficient = tall[k]

        # XXX HACK - CC coercion
        tcoefficient = CC(tcoefficient)
        def omega_gamma(s):
            s = CC(s)
            dtds = -tcoefficient
            val = omega_local(tcoefficient*(1-s)) * dtds
            return complex(val)
        return numpy.vectorize(omega_gamma, otypes=[complex])


####################################################
# Smale's Alpha Theory-based Riemann Surface Paths #
####################################################
ABELFUNCTIONS_SMALE_ALPHA0 = 1.1884471871911697 # = (13-2*sqrt(17))/4

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
    step = numpy.complex(1.0)
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


def smale_beta(df, xip1, yij):
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
    val : double
        :math:`\beta(f,x_{i+1},y_{i,j})`.
    """
    df0 = df[0]
    df1 = df[1]
    val = numpy.abs(df0(xip1,yij) / df1(xip1,yij))
    return val


def smale_gamma(df, xip1, yij):
    """Smale gamma function.

    Parameters
    ----------
    df : MultivariatePolynomial
        A list of all of the y-derivatives of f, including f itself.
    xip1 : complex
        The x-point to analytically continue to.
    yij : complex
        A y-root at xi. The root that we'll analytically continue.

    Returns
    -------
    double
        The Smale gamma function.
    """
    df0 = df[0]
    df1 = df[1]
    deg = len(df) - 1
    df1y = df1(xip1,yij)
    gamma = numpy.double(0)

    for n in range(2,deg+1):
        dfn = df[n]
        gamman = numpy.abs(dfn(xip1,yij) / (factorial(n)*df1y))
        gamman = gamman**(1.0/(n-1.0))
        if gamman > gamma:
            gamma = gamman
    return gamma


def smale_alpha(df, xip1, yij):
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
    double
        The Smale alpha function.
    """
    return smale_beta(df,xip1,yij) * smale_gamma(df,xip1,yij)



class RiemannSurfacePathSmale(RiemannSurfacePathPrimitive):
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
    def __init__(self, riemann_surface, complex_path, y0, ncheckpoints=16):
        # store a list of all y-derivatives of f (including the zeroth deriv)
        degree = riemann_surface.degree
        f = riemann_surface.f.change_ring(CC)
        x,y = f.parent().gens()
        df = [
            fast_callable(f.derivative(y,k), vars=(x,y), domain=complex)
            for k in range(degree+1)
        ]

        self.degree = degree
        self.df = df
        RiemannSurfacePathPrimitive.__init__(
            self, riemann_surface, complex_path, y0, ncheckpoints=ncheckpoints)

    def analytically_continue(self, xi, yi, xip1):
        # return the current fibre if the step size is too small
        if numpy.abs(xip1-xi) < 1e-14:
            return yi

        # first determine if the y-fibre guesses are 'approximate solutions'.
        # if any of them are not then refine the step by analytically
        # continuing to an intermediate "time"
        for j in range(self.degree):
            yij = yi[j]
            if smale_alpha(self.df, xip1, yij) > ABELFUNCTIONS_SMALE_ALPHA0:
                xiphalf = (xi + xip1)/2.0
                yiphalf = self.analytically_continue(xi, yi, xiphalf)
                yip1 = self.analytically_continue(xiphalf, yiphalf, xip1)
                return yip1

        # next, determine if the approximate solutions will converge to
        # different associated solutions
        for j in range(self.degree):
            yij = yi[j]
            betaij = smale_beta(self.df, xip1, yij)
            for k in range(j+1, self.degree):
                yik = yi[k]
                betaik = smale_beta(self.df, xip1, yik)
                distancejk = numpy.abs(yij-yik)
                if distancejk < 2*(betaij + betaik):
                    # approximate solutions don't lead to distinct roots.
                    # refine the step by analytically continuing to an
                    # intermedite time
                    xiphalf = (xi + xip1)/2.0
                    yiphalf = self.analytically_continue(xi, yi, xiphalf)
                    yip1 = self.analytically_continue(xiphalf, yiphalf, xip1)
                    return yip1

        # finally, since we know that we have approximate solutions that will
        # converge to difference associated solutions we will Netwon iterate
        yip1 = numpy.zeros(self.degree, dtype=complex)
        for j in range(self.degree):
            yip1[j] = newton(self.df, xip1, yi[j])
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
        return numpy.vectorize(omega_gamma, otypes=[complex])
