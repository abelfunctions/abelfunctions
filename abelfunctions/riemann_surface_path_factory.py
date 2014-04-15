"""
Riemann Surface Path Factory
============================

This module implements the :class:`RiemannSurfacePathFactory` class, a
class for generating :class:`RiemannSurfacePath` objects from various
kinds of input data.

Authors
-------

* Chris Swierczewski (January 2013)

"""

import numpy
import scipy
import sympy

from .analytic_continuation import (
    AnalyticContinuator,
    )
from .analytic_continuation_smale import (
    AnalyticContinuatorSmale,
)
from .riemann_surface_path import (
    RiemannSurfacePathPrimitive,
    RiemannSurfacePath,
    RiemannSurfacePathArc,
    RiemannSurfacePathLine,
    )
from .utilities import (
    Permutation,
    matching_permutation,
    )
from .xskeleton import XSkeleton
from .yskeleton import YSkeleton



class RiemannSurfacePathFactory(object):
    """The Path Factory for Riemann surfaces contains all of the methods
    needed to construct paths from place to place on the Riemann
    surface.

    The path factory makes heavy use of Monodromy and Homology to
    determine
    """
    def __init__(self, RS, kappa=3./5., base_point=None, base_sheets=None):
        self.RS = RS

        # initialize the X- and Y-skeletons at construction. this is
        # important! Also note that the monodromy group needs to be
        # computed from the X-Skeleton in order to compute the
        # Y-Skeleton
        self.XSkel = XSkeleton(RS, kappa=kappa, base_point=base_point)

        # use custom base sheets if provided. otherwise, numerically
        # compute a fixed ordering of sheets above the base poitn.
        if base_sheets:
            self._base_sheets = base_sheets
        else:
            p = RS.f.as_poly(RS.y)
            a = self.XSkel.base_point()
            coeffs = numpy.array(
                [c.evalf(subs={RS.x:a}, n=15) for c in p.all_coeffs()],
                dtype=numpy.complex)
            poly = numpy.poly1d(coeffs)
            self._base_sheets = (poly.r).tolist()

        # compute the y-skeleton
        self._monodromy_group = None
        self._monodromy_group = self.monodromy_group()
        self.YSkel = YSkeleton(RS, base_sheets=self._base_sheets,
                               monodromy_group=self._monodromy_group)

    def __str__(self):
        return 'Riemann Surface Path Factory for %s'%(self.RS)

    def base_point(self):
        return self.XSkel.base_point()

    def base_sheets(self):
        return self._base_sheets

    def base_place(self):
        return (self.base_point(), self.base_sheets())

    def branch_points(self):
        return self._monodromy_group[0]

    def discriminant_points(self):
        return self.XSkel.discriminant_points()

    def monodromy_group(self):
        """Returns the monodromy group of the algebraic curve defining the
        Riemann surface.

        Returns
        -------
        list of lists

            A list of two lists. The first is the list of branch points.
            The second is a list of corresponding permutation elements
            on the y-sheets of the curve.
        """
        if self._monodromy_group:
            return self._monodromy_group

        x0 = self.base_point()
        y0 = self.base_sheets()

        # compute the monodromy element for each discriminant point. if
        # it's the identity permutation, don't add to monodromy group
        branch_points = []
        permutations = []
        for bi in self.discriminant_points():
            # create the monodromy path
            xpath = self.XSkel.xpath_monodromy_path(bi)
            gamma = self.RiemannSurfacePath_from_xpath(xpath, x0, y0)

            # compute the end fibre and the corresponding permutation
            yend = gamma.get_y(1.0)
            phi = matching_permutation(y0, yend)

            # add the point to the monodromy group if the permutation is
            # not the identity.
            if not phi.is_identity():
                branch_points.append(bi)
                permutations.append(phi)

        # compute the monodromy element of the point at infinity
        xpath = self.XSkel.xpath_around_infinity()
        gamma = self.RiemannSurfacePath_from_xpath(xpath, x0, y0)
        yend = gamma.get_y(1.0)
        phi_oo = matching_permutation(y0, yend)
        if not phi_oo.is_identity():
            # sanity check: the product of the finite branch point
            # permutations should be equal to the inverse of the
            # permutation at infinity
            phi_prod = reduce(lambda phi1,phi2: phi2*phi1, permutations)
            phi_prod = phi_prod * phi_oo
            if phi_prod.is_identity():
                branch_points.append(sympy.oo)
                permutations.append(phi_oo)
            else:
                raise ValueError('Contradictory permutation at infinity.')

        return branch_points, permutations

    def monodromy_path(self, bi):
        """Returns the monodromy path around the discriminant point `bi`.

        Arguments
        ---------
        bi : complex
            A discriminant point of the curve.
        nrots : optional, integer
            The number of times to rotate around `bi`. (Default 1).

        Returns
        -------
        RiemannSurfacePath
            The path encircling the branch point `bi`.

        """
        xpath = self.XSkel.xpath_monodromy_path(bi)
        gamma = self.RiemannSurfacePath_from_xpath(xpath)
        return gamma

    def a_cycles(self):
        """Returns the a-cycles on the Riemann surface.

        Returns
        -------
        list, RiemannSurfacePath
            A list of the a-cycles of the Riemann surface as
            :class:`RiemannSurfacePath` objects.
        """
        cycles = self.YSkel.a_cycles()
        paths = map(self.RiemannSurfacePath_from_cycle, cycles)
        return paths

    def b_cycles(self):
        """Returns the b-cycles on the Riemann surface.

        Returns
        -------
        list, RiemannSurfacePath
            A list of the b-cycles of the Riemann surface as
            :class:`RiemannSurfacePath` objects.
        """
        cycles = self.YSkel.b_cycles()
        paths = map(self.RiemannSurfacePath_from_cycle, cycles)
        return paths

    def c_cycles(self):
        """Returns the c-cycles of the Riemann surface and the linear
        combination matrix defining the a- and b-cycles from the c-cycles.

        The a- and b- cycles of the Riemann surface are formed from
        linear combinations of the c-cycles. These linear combinations
        are obtained from the :method:`linear_combinations` method.

        .. note::

            It may be computationally more efficient to integrate over
            the (necessary) c-cycles and take linear combinations of the
            results than to integrate over the a- and b-cycles
            separately. Sometimes the column rank of the linear
            combination matrix (that is, the number of c-cycles used to
            construct a- and b-cycles) is lower than the size of the
            homology group and sometimes the c-cycles are simpler and
            shorter than the homology cycles.

        Returns
        -------
        list, RiemannSurfacePath
            A list of the c-cycles of the Riemann surface as
            :class:`RiemannSurfacePath` objects.

        """
        # prune the linear combinations matrix of the columns that don't
        # contribute to a c-cycles
        cycles, linear_combinations = self.YSkel.c_cycles()
        ncols = linear_combinations.shape[1]
        indices = [j for j in range(ncols)
                   if not (linear_combinations[:,j] == 0).all()]
        c_cycles = [cycles[i] for i in indices]
        linear_combinations = linear_combinations[:,indices]

        paths = map(self.RiemannSurfacePath_from_cycle, c_cycles)
        return paths, linear_combinations

    def RiemannSurfacePath_from_cycle(self, cycle):
        """Constructs a :class:`RiemannSurfacePath` object from a list of
        x-path data.

        Arguments
        ---------
        ypath : list
            A list fo tuples defining a y-path. The output of
            `self.a_cycles()`, `self.b_cycles()`, etc.
        """
        xpath = []
        for bi, nrots in cycle:
            xpath.extend(self.XSkel.xpath_monodromy_path(bi, nrots=nrots))
        return self.RiemannSurfacePath_from_xpath(xpath)

    def RiemannSurfacePath_from_xpath(self, xpath, x0=None, y0=None):
        """Constructs a :class:`RiemannSurfacePath` object from a list of
        x-path data.

        Arguments
        ---------
        xpath : list
            A list of tuples defining the xpath.
        x0 : complex (default `self.base_point()`)
            The starting x-point of the path.
        y0 : complex list (default `self.base_sheets()`)
            The starting ordering of the y-sheets.

        Returns
        -------
        RiemannSurfacePath
            A path on the Riemann surface with the prescribed x-path.

        """
        if x0 is None:
            x0 = self.base_point()
        if y0 is None:
            y0 = self.base_sheets()

        ACSmale = AnalyticContinuatorSmale(self.RS)
        # ACPuisuex = AnalyticContinuatorPuiseux(self.RS)
        x0 = numpy.complex(x0)
        y0 = numpy.array(y0, dtype=numpy.complex)
        x0_segment = x0
        y0_segment = y0
        segments = []
        for data in xpath:
            # pick an analytic continuator (XXX PROVIDE LOGIC FOR THIS XXX)
            AC = ACSmale

            # build a segment based on the xpath data
            if len(data) == 2:
                segment = RiemannSurfacePathLine(self.RS, AC, x0_segment,
                                                 y0_segment, *data)
            else:
                segment = RiemannSurfacePathArc(self.RS, AC, x0_segment,
                                                y0_segment, *data)

            # determine the starting place of the next segment
            x0_segment = segment.get_x(1)
            y0_segment = segment.get_y(1)
            segments.append(segment)

        # build the entire path from the path segments
        segments = numpy.array(segments, dtype=RiemannSurfacePathPrimitive)
        gamma = RiemannSurfacePath(self.RS, x0, y0, segments)
        return gamma
