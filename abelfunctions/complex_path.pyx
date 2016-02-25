#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False

r"""Complex Paths :mod:`abelfunctions.complex_path`

Data structures for paths in the complex plane.

Classes
-------

.. autosummary::

  ComplexPathPrimitive
  ComplexPath
  ComplexPathLine
  ComplexPathArc
  ComplexPathRay

Contents
--------
"""

import numpy
cimport numpy

from sage.functions.other import real_part, imag_part
from sage.plot.line import line

cdef extern from 'math.h':
    int floor(double)

cdef extern from 'complex.h':
    complex cexp(complex)
    double cabs(complex)

cdef class ComplexPathPrimitive:
    r"""
    Base class for paths in the complex plane.
    """
    @property
    def segments(self):
        return [self]

    ##############################
    # overload these in subclass #
    ##############################
    def __init__(self, *args, **kwds):
        self._nsegments = 1

    def __repr__(self):
        start_point = self[0](0.0)
        end_point = self[self._nsegments-1](1.0)
        s = 'Complex path from %s to %s'%(start_point, end_point)
        return s

    cpdef complex eval(self, double s):
        raise NotImplementedError('Implement in subclass.')

    cpdef complex derivative(self, double s):
        raise NotImplementedError('Implement in subclass.')

    def reverse(self):
        raise NotImplementedError('Implement in subclass.')

    ######################
    # additional methods #
    ######################
    def __add__(self, other):
        # assert that the path is continuous
        self_end = self(1.0)
        other_start = other(0.0)
        eps = 1e-12
        if abs(self_end - other_start) > eps:
            raise ValueError('Cannot form sum of complex paths: ending point '
                             'of left does not match start point of right.')

        # form the complex path
        segments = self.segments + other.segments
        gamma = ComplexPath(segments)
        return gamma

    def __call__(self, s):
        cdef int j, N
        cdef double[:] inputs
        cdef complex[:] values

        # if the input is an array then amortize the calculation
        if isinstance(s, numpy.ndarray):
            N = len(s)
            inputs = s.astype(numpy.double)
            values = numpy.zeros(len(s), dtype=complex)
            for j in range(N):
                values[j] = self.eval(inputs[j])
            return numpy.array(values, dtype=complex)

        # otherwise, just evaluate at the point
        cdef complex value = self.eval(s)
        return value

    def plot(self, plot_points=128, **kwds):
        r"""Return a plot of the path.

        Parameters
        ----------
        plot_points : int or list
            The number or plot points or a list of parameter values lying in
            the interval [0,1]. (Default: 128)
        **kwds : dict
            Additional keywords passed to `sage.plot.line.line`.

        Returns
        -------
        plt : Sage plot
            A plot of the complex path.
        """
        # if the plot_points are given as a list then use the list of
        # parameters. otherwise, create a linspace
        s = plot_points
        if not (isinstance(s, list) or isinstance(s, numpy.ndarray)):
            s = numpy.linspace(0,1,s)

        # s is now a list of points. compute the path points and draw
        vals = [self(si) for si in s]
        pts = [(real_part(x), imag_part(x)) for x in vals]
        plt = line(pts, **kwds)
        return plt


cdef class ComplexPath(ComplexPathPrimitive):
    r"""
    A composite path in the complex plane.

    Every `ComplexPath` is composed of individual primitive paths, called
    "segments". Every path is parameterized from `s=0` to `s=1`. `ComplexPath`
    follows the composite design pattern.

    Attributes
    ----------
    segments : list
        A list of the constituent segments of the path.

    """
    @property
    def segments(self):
        return numpy.asarray(self._segments).tolist()

    def __init__(self, segments):
        r"""Directly instantiate an ComplexPath composite from a list of
        ComplexPathPrimitives.

        Parameters
        ----------
        *args : list
            A list of :class:`ComplexPathPrimitive`s.
        """
        ComplexPathPrimitive.__init__(self)

        # assert that the segments form a continuous path
        cdef ComplexPathPrimitive[:] args = numpy.array(
            segments, dtype=ComplexPathPrimitive)
        n = len(segments)
        eps = 1e-12
        for k in range(n-1):
            gamma0 = segments[k]
            gamma1 = segments[k+1]
            if abs(gamma1(0.0) - gamma0(1.0)) > eps:
                raise ValueError('Segments must form continuous path.')

        self._segments = numpy.array(segments, dtype=ComplexPathPrimitive)
        self._nsegments = n

    def __getitem__(self, index):
        r"""Return the segment at index `index`"""
        return self.segments[index]

    cdef int segment_index_at_parameter(self, double s):
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
        cdef int k = floor(s*self._nsegments)
        cdef int diff = (self._nsegments - 1) - k
        cdef int dsgn = diff >> 31
        cdef int index = k + (diff & dsgn)
        return index

    cpdef complex eval(self, double s):
        r"""Return the complex point along the path at the parameter `s`.

        .. note::

            Directly called by :meth:`__call__`.

        Parameters
        ----------
        s : float
            Path parameter in the interval [0,1].

        Returns
        -------
        val : complex
            The point
        """
        cdef int k = self.segment_index_at_parameter(s)
        cdef double s_seg = s*self._nsegments - k
        cdef ComplexPathPrimitive seg = self._segments[k]
        cdef complex val = seg.eval(s_seg)
        return val

    cpdef complex derivative(self, double s):
        r"""Return the derivative of the complex path with respect to the
        parameter.

        Parameters
        ----------
        s : float
            Path parameter in the interval [0,1].

        Returns
        -------
        index : int
            The index `k` of the path segment :math:`\gamma_k`.
        """
        cdef int k = self.segment_index_at_parameter(s)
        cdef double s_seg = s*self._nsegments - k
        cdef ComplexPathPrimitive seg = self._segments[k]
        cdef complex val = seg.derivative(s_seg)
        return val

    def reverse(self):
        r"""Return the path reversed.

        Parameters
        ----------
        None

        Returns
        -------
        gamma : ComplexPath
        """
        reversed_segments = [s.reverse() for s in self.segments[::-1]]
        gamma = ComplexPath(reversed_segments)
        return gamma

    def plot(self, plot_points=128, **kwds):
        r"""Return a plot of the path.

        Parameters
        ----------
        plot_points : int or list
            The number or plot points or a list of parameter values lying in
            the interval [0,1]. (Default: 128)
        **kwds : dict
            Additional keywords passed to `sage.plot.line.line`.

        Returns
        -------
        plt : Sage plot
            A plot of the complex path.
        """
        # if explicit points are given then plot as usual
        if (isinstance(plot_points, list) or
            isinstance(plot_points, numpy.ndarray)):
            return ComplexPathPrimitive.plot(self, plot_points, **kwds)

        # otherwise, plot one segment at a time so as to include the endpoints
        # of each segment (otherwise, it looks fragmented)
        s_seg = floor(plot_points / self._nsegments)
        plt = sum(seg.plot(s_seg, **kwds) for seg in self._segments)
        return plt

cdef class ComplexLine(ComplexPathPrimitive):
    r"""A line segment in the complex plane.

    Attributes
    ----------
    x0 : complex
        The starting point of the line.
    x1 : complex
        The ending point of the line.
    """
    @property
    def x0(self):
        return self._x0

    @property
    def x1(self):
        return self._x1

    def __init__(self, complex x0, complex x1):
        ComplexPathPrimitive.__init__(self)
        self._x0 = x0
        self._x1 = x1

    def __repr__(self):
        s = 'Line(%s,%s)'%(self.x0, self.x1)
        return s

    def __richcmp__(self, other, int op):
        if not isinstance(other, ComplexLine):
            return 3
        if (self.x0 == other.x0) and (self.x1 == other.x1):
            return 2
        return 3

    cpdef complex eval(self, double s):
        cdef complex val = self._x0 + (self._x1-self._x0)*s
        return val

    cpdef complex derivative(self, double s):
        cdef complex val = self._x1 - self._x0
        return val

    def reverse(self):
        return ComplexLine(self._x1, self._x0)


cdef class ComplexArc(ComplexPathPrimitive):
    r"""A complex arc. (Part of a circle in the complex plane.)

    Attributes
    ----------
    R : complex
        The radius of the arc.
    w : complex
        The center of the arc.
    theta : complex
        The starting angle (in radians) on the arc. Usually 0 or :math:`\pi`.
    dtheta : complex
        The number of radians to travel where the sign of `dtheta`
        indicates direction. The absolute value of `dtheta` is equal to
        the arc length.
    """
    @property
    def R(self):
        return self._R

    @property
    def w(self):
        return self._w

    @property
    def theta(self):
        return self._theta

    @property
    def dtheta(self):
        return self._dtheta

    def __init__(self, double R, complex w, double theta, double dtheta):
        ComplexPathPrimitive.__init__(self)
        self._R = R
        self._w = w
        self._theta = theta
        self._dtheta = dtheta

    def __repr__(self):
        s = 'Arc(%s,%s,%s,%s)'%(self._R, self._w, self._theta, self._dtheta)
        return s

    def __richcmp__(self, other, int op):
        if not isinstance(other, ComplexArc):
            return 3
        if ((self.R == other.R) and (self.w == other.w) and
            (self.theta == other.theta) and (self.dtheta == other.dtheta)):
            return 2
        return 3

    cpdef complex eval(self, double s):
        cdef complex val = \
            self._R*cexp(1.0j*(self._theta + s*self._dtheta)) + self._w
        return val

    cpdef complex derivative(self, double s):
        cdef complex val = (self._R*1.0j*self._dtheta) * \
            cexp(1.0j*(self._theta + s*self._dtheta))
        return val

    def reverse(self):
        return ComplexArc(
            self._R, self._w, self._theta+self._dtheta, -self._dtheta)


cdef class ComplexRay(ComplexPathPrimitive):
    r"""A complex ray: a path with a finite starting point going to infinity.

    Attributes
    ----------
    x0 : complex
        The starting point of the ray.
    """
    @property
    def x0(self):
        return self._x0

    def __init__(self, complex x0):
        ComplexPathPrimitive.__init__(self)
        if cabs(x0) < 1e-8:
            raise ValueError('Complex rays must start away from the origin.')
        self._x0 = x0

    def __repr__(self):
        s = 'Arc(%s)'%self._x0
        return s

    def __richcmp__(self, ComplexRay other, int op):
        if not isinstance(other, ComplexRay):
            return 3
        if self.x0 != other.x0:
            return 3
        return 2

    cpdef complex eval(self, double s):
        if s == 1.0: return float('inf')
        cdef complex val = self._x0/(1-s)
        return val

    cpdef complex derivative(self, double s):
        if s == 1.0: return float('inf')
        cdef complex val = -self._x0/(1.-s)**2
        return val

    def reverse(self):
        raise ValueError('Cannot reverse paths to infinity.')
