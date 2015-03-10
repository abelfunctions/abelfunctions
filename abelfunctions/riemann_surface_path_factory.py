"""
Riemann Surface Path Factory
============================

This module implements the :class:`RiemannSurfacePathFactory` class, a
class for generating :class:`RiemannSurfacePath` objects from various
kinds of input data.

Classes
-------

RiemannSurfacePathFactory

Functions
---------


Authors
-------

* Chris Swierczewski (January 2013)

"""

import pdb

import numpy
import scipy
import scipy.linalg as linalg
import sympy

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
from .xpath_factory import XPathFactory, XPathFactoryAbel
from .ypath_factory import YPathFactory

class RiemannSurfacePathFactory(object):
    r"""Factory class for constructing paths on the Riemann surface.

    Attributes
    ----------
    RS : RiemannSurface
        The Riemann surface.
    XPF : XPathFactory
        A factory object for determining how to navigate the complex
        x-plane. How to construct paths avoiding, encircling, and
        approaching discriminant points.
    YPF : YPathFactory
        A factory object for determining how to navigate the complex
        y-plane. Computes the branching structure and homology of the
        surface. Defines how to swap sheets.
    _base_sheets : list
        The ordered sheets of the surface at the base point.
    _monodromy_group : list
        The monodromy group of the curve. Used to compute the homology.

    Methods
    -------
    base_point
    base_sheets
    base_place
    discriminant_points
    closest_discriminant_point
    branch_points
    path_to_place
    monodromy_group
    monodromy_path
    a_cycles
    b_cycles
    c_cycles
    show_paths
    RiemannSurfacePath_from_cycle
    RiemannSurfacePath_from_xpath
    _path_to_discriminant_place
    _path_to_regular_place
    """
    def __init__(self, RS, kappa=2./5.):
        r"""Initialize the Riemann surface path factory.

        The user can manually supply a base point in the complex x-plane
        as well as an ordering of the base sheets at the base point. The
        base point must be a regular point of the curve.

        A "relaxation parameter" `kappa` is used when determining the
        radii of the "bounding circles" around each discriminant point
        of the curve. Paths in the complex plane inside of these
        bounding circles need to be constructed with care since the
        y-roots of the curve are (possibly) beginning to converge to one
        another.

        Parameters
        ----------
        RS : RiemannSurface
            The Riemann surface.
        kappa : double
            A relaxation factor between 0 and 1.
        base_point : complex
            An optional user-defined base x-point for the mondromy group
            and Riemann surface.
        base_sheets : complex, list
            An optional user-defined ordered list of y-roots above the
            base point.

        Returns
        -------
        None
        """
        self.RS = RS
        self._base_point = self.RS.base_point()
        self._base_sheets = self.RS.base_sheets()

        # initialize the X- and Y-PathFactories at construction. Note
        # that the monodromy group needs to be computed from the
        # XPathFactory in order to compute the YPathFactory
        self.XPF = XPathFactoryAbel(RS, kappa=kappa,
                                    base_point=self._base_point)
        # compute the y-skeleton
        self._monodromy_group = None
        self._monodromy_group = self.monodromy_group()
        self.YPF = YPathFactory(self.RS, base_sheets=self._base_sheets,
                                monodromy_group=self._monodromy_group)

    def __str__(self):
        return 'Riemann Surface Path Factory for %s'%(self.RS)

    def show_paths(self, ax=None, *args, **kwds):
        r"""Plots all of the monodromy paths of the curve.

        Arguments
        ---------
        ax : matplotlib.Axes
            The figure axes on which to plot the paths.

        Returns
        -------
        None
        """
        return self.XPF.show_paths(ax=ax, *args, **kwds)

    def base_point(self):
        return self.XPF.base_point()

    def base_sheets(self):
        return self._base_sheets

    def base_place(self):
        return (self.base_point(), self.base_sheets())

    def branch_points(self):
        return self._monodromy_group[0]

    def discriminant_points(self, exact=False):
        r"""Same as :func:`RiemannSurface.discriminant_points`."""
        return self.RS.discriminant_points(exact=exact)

    def closest_discriminant_point(self, x, exact=True):
        r"""Same as :func:`RiemannSurface.closest_discriminant_point`."""
        return self.RS.closest_discriminant_point(x, exact=exact)

    def path_to_place(self, P):
        r"""Returns a path to a specified place `P` on the Riemann surface.

        Parameters
        ----------
        P : complex tuple
            A place on the Riemann surface.
        """
        if P.is_discriminant():
            return self._path_to_discriminant_place(P)
        else:
            return self._path_to_regular_place(P)

    def _path_to_discriminant_place(self, P):
        r"""Returns a path to a discriminant place on the surface.

        A "discriminant" place :math:`P` is a place on the Riemann
        surface where Puiseux series are required to determine the x-
        and y-projections of the place.

        Parameters
        ----------
        P : Place

        Returns
        -------
        RiemannSurfacePath

        """
        # compute a valid y-value at x=a=b-R, where b is the
        # discriminant point center of the series and R is the radius of
        # bounding circle around b, such that analytically continuing
        # from that (x,y) to x=b will reach the designated place
        p = P.puiseux_series
        center, coefficient, ramification_index = p.xdata
        R = self.XPF.radius(center)
        a = center - R
        t = (-R/coefficient)**(1.0/ramification_index)
        # p.coerce_to_numerical()
        p.extend_to_t(t)
        y = p.eval_y(t)

        # construct a path going from the base place to this regular
        # place to the left of the target discriminant point
        P1 = self.RS(a,y)
        gamma1 = self._path_to_regular_place(P1)

        # construct the RiemannSurfacePath going from this regular place
        # to the discriminant point.
        xend = gamma1.get_x(1.0)
        yend = gamma1.get_y(1.0)
        xdata = (a, center)
        segment = RiemannSurfacePathLine(self.RS, xend, yend, *xdata)
        gamma = gamma1 + segment
        return gamma

    def _path_to_regular_place(self, P):
        r"""
        Returns a path to a regular place on the surface.

        A "regular" place :math:`P` is a place on the Riemann surface
        where the x-projection of the place is not a discriminant point
        of the curve.

        Parameters
        ----------
        P : Place

        Returns
        -------
        RiemannSurfacePath

        """
        # construct the reverse path from the place to the base x-point
        # in order to determine which sheet the place is on
        P0 = self.base_place()
        xpath_to_target = self.XPF.xpath_build_avoiding_path(P0[0],P.x)
        gamma = self.RiemannSurfacePath_from_xpath(xpath_to_target)

        # determine the index of the sheet (sheet_index) of the base
        # place continues to the y-component of the target
        base_sheets = self.base_sheets()
        end_sheets = numpy.array(gamma.get_y(1.0), dtype=numpy.complex)
        end_diffs = numpy.abs(end_sheets - P.y)
        if numpy.min(end_diffs) > 1.0e-8:
            raise ValueError('Error in constructing Abel path: end of regular '
                             'path does not match with target place.')
        sheet_index = numpy.argmin(end_diffs)

        # if this sheet index is zero (the base sheet) then simply
        # return the constructed path gamma. otherwise, construct the
        # branch switching path
        if sheet_index != 0:
            ypath = self.YPF.ypath_from_base_to_sheet(sheet_index)
            gamma_swap = self.RiemannSurfacePath_from_cycle(ypath)
            ordered_base_sheets = gamma_swap.get_y(1.0)

            # take the new ordering of branch points and construct the
            # path to the target place. there is an inherent check that
            # the y-values above the base point match
            gamma_rest = self.RiemannSurfacePath_from_xpath(
                xpath_to_target, y0=ordered_base_sheets)
            gamma = gamma_swap + gamma_rest

        # sanity check
        yend = gamma.get_y(1.0)[0]
        if numpy.abs(yend - P.y) > 1.0e-8:
            raise ValueError('Error in constructing Abel path.')
        return gamma

    def _path_near_discriminant_point(self, P):
        """Returns a path to a place `P` where the x-coordinate is a
        discriminant point on the surface.
        """
        raise NotImplementedError()

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
            xpath = self.XPF.xpath_monodromy_path(bi)
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
        xpath = self.XPF.xpath_around_infinity()
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
        xpath = self.XPF.xpath_monodromy_path(bi)
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
        cycles = self.YPF.a_cycles()
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
        cycles = self.YPF.b_cycles()
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
        cycles, linear_combinations = self.YPF.c_cycles()
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
            xpath.extend(self.XPF.xpath_monodromy_path(bi, nrots=nrots))
        return self.RiemannSurfacePath_from_xpath(xpath)

    def RiemannSurfacePath_from_xpath(self, xpath, x0=None, y0=None):
        r"""Constructs a :class:`RiemannSurfacePath` object from x-path data.

        Parameters
        ----------
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

        # build a list of path segments from each tuple of xdata. build
        # a line segment or arc depending on xdata input
        x0 = numpy.complex(x0)
        y0 = numpy.array(y0, dtype=numpy.complex)
        x0_segment = x0
        y0_segment = y0
        segments = []
        for data in xpath:
            if len(data) == 2:
                segment = RiemannSurfacePathLine(self.RS, x0_segment,
                                                 y0_segment, *data)
            else:
                segment = RiemannSurfacePathArc(self.RS, x0_segment,
                                                y0_segment, *data)

            # determine the starting place of the next segment
            x0_segment = segment.get_x(1.0)
            y0_segment = segment.get_y(1.0)
            segments.append(segment)

        # build the entire path from the path segments
        segments = numpy.array(segments, dtype=RiemannSurfacePathPrimitive)
        gamma = RiemannSurfacePath(self.RS, x0, y0, segments)
        return gamma
