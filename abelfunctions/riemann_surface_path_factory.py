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

import numpy

from abelfunctions.complex_path import (
    ComplexLine,
    ComplexRay,
    ComplexPath,
)
from abelfunctions.divisor import RegularPlace
from abelfunctions.riemann_surface_path import (
    RiemannSurfacePath,
    RiemannSurfacePathPuiseux,
    RiemannSurfacePathSmale,
    )
from abelfunctions.utilities import matching_permutation
from abelfunctions.complex_path_factory import ComplexPathFactory
#from abelfunctions.skeleton import Skeleton
from abelfunctions.ypath_factory import YPathFactory as Skeleton

from numpy import complex, array
from sage.all import infinity, CC, CDF, cached_method

class RiemannSurfacePathFactory(object):
    r"""Factory class for constructing paths on the Riemann surface.

    Attributes
    ----------
    RS : RiemannSurface
        The Riemann surface.
    complex_path_factory : ComplexPathFactory
        A factory object for determining how to navigate the complex
        x-plane. How to construct paths avoiding, encircling, and
        approaching discriminant points.
    skeleton : Skeleton
        An object for determining how to navigate the complex y-plane. Computes
        the branching structure and homology of the surface. Defines how to
        swap sheets.
    _base_sheets : list
        The ordered sheets of the surface at the base point.
    _monodromy_group : list
        The monodromy group of the curve. Used to compute the homology.

    Methods
    -------
    .. autosummary::

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
      RiemannSurfacePath_from_complex_path
      _path_to_discriminant_place
      _path_to_regular_
    """
    @property
    def base_point(self):
        return self.complex_path_factory.base_point

    @property
    def base_sheets(self):
        return self._base_sheets

    @property
    def base_place(self):
        return RegularPlace(self, self.base_point, self.base_sheets[0])

    @property
    def branch_points(self):
        return self.monodromy_group()[0]

    @property
    def discriminant_points(self):
        return self.complex_path_factory.discriminant_points

    @property
    def skeleton(self):
        if not self._skeleton:
            mon = (self.base_point, self.base_sheets) + \
                  self.monodromy_group()
            self._skeleton = Skeleton(self.riemann_surface, mon)
        return self._skeleton

    def __init__(self, riemann_surface, base_point=None, base_sheets=None,
                 kappa=3./5.):
        r"""Initialize the Riemann surface path factory.

        The user can manually supply a base point in the complex x-plane as
        well as an ordering of the base sheets at the base point. The base
        point must be a regular point of the curve.

        A "relaxation parameter" `kappa` is used when determining the radii of
        the "bounding circles" around each discriminant point of the curve.
        Paths in the complex plane inside of these bounding circles need to be
        constructed with care since the y-roots of the curve are (possibly)
        beginning to converge to one another.

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
        self.riemann_surface = riemann_surface
        self.complex_path_factory = ComplexPathFactory(
            riemann_surface.f, base_point=base_point, kappa=kappa)

        # now that the compelx path factory is built we have a base point
        # (either provided or chosen). we now need to determine the base
        # sheets.
        #
        # if not provided then compute some. otherwise, check they are roots
        if not base_sheets:
            x,y = self.riemann_surface.f.parent().gens()
            p = self.riemann_surface.f(self.base_point, y).univariate_polynomial()
            roots = p.roots(CDF, multiplicities=False)
            base_sheets = numpy.array(roots, dtype=numpy.complex)
        else:
            f = self.riemann_surface.f
            for sheet in base_sheets:
                value = f(self.base_point, sheet)
                if abs(value) > 1e-15:
                    raise ValueError('Base sheets %s do not lie above base '
                                     'point %s'%(base_sheets, self.base_point))
        self._base_sheets = base_sheets

        # skeleton is computed when first used
        self._skeleton = None

    def __repr__(self):
        s = 'Path Factory for %s'%(self.riemann_surface)
        return s

    def show_paths(self, *args, **kwds):
        r"""Plots all of the monodromy paths of the curve.

        Parameters
        ----------
        args : list
            See :func:`ComplexPathFactory.show_paths`
        kwds : dict
            See :func:`ComplexPathFactory.show_paths`

        Returns
        -------
        plt : Sage plot
            A plot of the monodromy paths.
        """
        plt = self.complex_path_factory.show_paths(*args, **kwds)
        return plt

    def closest_discriminant_point(self, x, **kwds):
        r"""Returns the discriminant point closest to `x`.

        See Also
        --------
        :func:`ComplexPathFactory.closest_discriminant_point`

        """
        b = self.complex_path_factory.closest_discriminant_point(x)
        return b

    def path_to_place(self, P):
        r"""Returns a path to a specified place `P` on the Riemann surface.

        Parameters
        ----------
        P : complex tuple
            A place on the Riemann surface.

        See Also
        --------
        :func:`PathFactory._path_to_infinite_place`
        :func:`PathFactory._path_to_discriminant_place`
        :func:`PathFactory._path_to_regular_place`
        """
        if P.is_infinite():
            return self._path_to_infinite_place(P)
        elif P.is_discriminant():
            return self._path_to_discriminant_place(P)
        else:
            return self._path_to_regular_place(P)

    def _path_to_infinite_place(self, P):
        r"""Returns a path to a place at an infintiy of the surface.

        A place at infinity is one where the `x`-projection of the place is the
        point `x = \infty` of the complex Riemann sphere. An infinite place is
        a type of discirminant place.

        Parameters
        ----------
        P : Place
            The target infinite place.

        Returns
        -------
        gamma : RiemannSurfacePath
            A path from the base place to the place at infinity.

        """
        # determine a place Q from where we can reach the target place P. first,
        # pick an appropriate x-point over which a Q is chosen
        x0 = self.base_point
        if numpy.real(x0) < 0:
            xa = 5*x0  # move away from the origin
        else:
            xa = -5   # arbitrary choice away from the origin

        # next, determine an appropriate y-part from where we can reach the
        # place at infinity
        p = P.puiseux_series
        center, coefficient, ramification_index = p.xdata
        ta = CC(xa/coefficient).nth_root(abs(ramification_index))
        ta = ta if ramification_index > 0 else 1/ta
        p.extend_to_t(ta)
        ya = complex(p.eval_y(ta))

        # construct the place Q and compute the path going from P0 to Q
        Q = self.riemann_surface(xa,ya)
        gamma_P0_to_Q = self.path_to_place(Q)

        # construct the path going from Q to P
        xend = complex(gamma_P0_to_Q.get_x(1.0))
        yend = array(gamma_P0_to_Q.get_y(1.0), dtype=complex)
        gamma_x = ComplexRay(xend)
        gamma_Q_to_P = RiemannSurfacePathPuiseux(
            self.riemann_surface, gamma_x, yend)
        gamma = gamma_P0_to_Q + gamma_Q_to_P
        return gamma


    def _path_to_discriminant_place(self, P):
        r"""Returns a path to a discriminant place on the surface.

        A "discriminant" place :math:`P` is a place on the Riemann
        surface where Puiseux series are required to determine the x-
        and y-projections of the place.

        Parameters
        ----------
        P : Place
            A place on the Riemann surface whose x-projection is a discriminant
            point of the curve.

        Returns
        -------
        gamma : RiemannSurfacePath
            A path from the base place to the discriminant place `P`.

        """
        # compute a valid y-value at x=a=b-R, where b is the
        # discriminant point center of the series and R is the radius of
        # bounding circle around b, such that analytically continuing
        # from that (x,y) to x=b will reach the designated place
        p = P.puiseux_series
        center, coefficient, ramification_index = p.xdata
        R = self.complex_path_factory.radius(center)
        a = center - R
        t = (-R/coefficient)**(1.0/ramification_index)
        # p.coerce_to_numerical()
        p.extend_to_t(t)
        y = p.eval_y(t)

        # construct a path going from the base place to this regular
        # place to the left of the target discriminant point
        P1 = self.riemann_surface(a,y)
        gamma1 = self._path_to_regular_place(P1)

        # construct the RiemannSurfacePath going from this regular place to the
        # discriminant point.
        yend = array(gamma1.get_y(1.0), dtype=complex)
        gamma_x = ComplexLine(complex(a), complex(center))
        segment = RiemannSurfacePathPuiseux(
            self.riemann_surface, gamma_x, yend)
        gamma = gamma1 + segment
        return gamma

    def _path_to_regular_place(self, P):
        r"""Returns a path to a regular place on the surface.

        A "regular" place :math:`P` is a place on the Riemann surface
        where the x-projection of the place is not a discriminant point
        of the curve.

        Parameters
        ----------
        P : Place
            A place on the Riemann surface whose x-projection is a regular
            point of the curve.

        Returns
        -------
        gamma : RiemannSurfacePath
            A path from the base place to the regular place `P`.

        """
        # construct the reverse path from the place to the base x-point in
        # order to determine which sheet the place is on
        P0 = self.base_place
        gamma_x = self.complex_path_factory.path(P0.x, P.x)
        gamma = self.RiemannSurfacePath_from_complex_path(gamma_x)

        # determine the index of the sheet (sheet_index) of the base place
        # continues to the y-component of the target
        end_sheets = array(gamma.get_y(1.0), dtype=complex)
        end_diffs = abs(end_sheets - complex(P.y))
        if min(end_diffs) > 1.0e-8:
            raise ValueError('Error in constructing Abel path: end of regular '
                             'path does not match with target place.')
        sheet_index = numpy.argmin(end_diffs)

        # if this sheet index is zero (the base sheet) then simply return the
        # constructed path gamma. otherwise, construct the branch switching
        # path
        if sheet_index != 0:
            ypath = self.skeleton.ypath_from_base_to_sheet(sheet_index)
            gamma_swap = self.RiemannSurfacePath_from_cycle(ypath)
            ordered_base_sheets = numpy.array(gamma_swap.get_y(1.0), dtype=complex)

            # take the new ordering of branch points and construct the
            # path to the target place. there is an inherent check that
            # the y-values above the base point match
            gamma_rest = self.RiemannSurfacePath_from_complex_path(
                gamma_x, y0=ordered_base_sheets)
            gamma = gamma_swap + gamma_rest

        # sanity check
        yend = gamma.get_y(1.0)[0]
        if numpy.abs(yend - numpy.complex(P.y)) > 1.0e-8:
            raise ValueError('Error in constructing Abel path.')
        return gamma

    def _path_near_discriminant_point(self, P):
        """Returns a path to a place `P` where the x-coordinate is a
        discriminant point on the surface.
        """
        raise NotImplementedError()

    @cached_method
    def monodromy_group(self):
        """Returns the monodromy group of the algebraic curve defining the
        Riemann surface.

        Returns
        -------
        list : list
            A list containing the base point, base sheets, branch points, and
            corresponding monodromy group elements.

        """
        x0 = self.base_point
        y0 = self.base_sheets

        # compute the monodromy element for each discriminant point. if
        # it's the identity permutation, don't add to monodromy group
        branch_points = []
        permutations = []
        for bi in self.discriminant_points:
            gamma = self.monodromy_path(bi)

            # compute the end fibre and the corresponding permutation
            yend = array(gamma.get_y(1.0), dtype=complex)
            phi = matching_permutation(y0, yend)

            # add the point to the monodromy group if the permutation is
            # not the identity.
            if not phi.is_identity():
                branch_points.append(bi)
                permutations.append(phi)

        # compute the monodromy element of the point at infinity
        gamma_x = self.complex_path_factory.monodromy_path_infinity()
        gamma = self.RiemannSurfacePath_from_complex_path(gamma_x, x0, y0)
        yend = array(gamma.get_y(1.0), dtype=complex)
        phi_oo = matching_permutation(y0, yend)

        # sanity check: the product of the finite branch point permutations
        # should be equal to the inverse of the permutation at infinity
        phi_prod_all = phi_oo
        for sigma in permutations:
            phi_prod_all = sigma * phi_prod_all
        if not phi_prod_all.is_identity():
            raise ValueError('Contradictory permutation at infinity.')

        # add infinity if it's a branch point
        if not phi_oo.is_identity():
            branch_points.append(infinity)
            permutations.append(phi_oo)

        return branch_points, permutations

    def monodromy_path(self, bi, nrots=1):
        """Returns the monodromy path around the discriminant point `bi`.

        Parameters
        ----------
        bi : complex
            A discriminant point of the curve.
        nrots : optional, integer
            The number of times to rotate around `bi`. (Default 1).

        Returns
        -------
        RiemannSurfacePath
            The path encircling the branch point `bi`.

        """
        gammax = self.complex_path_factory.monodromy_path(bi, nrots=nrots)
        gamma = self.RiemannSurfacePath_from_complex_path(gammax)
        return gamma

    def a_cycles(self):
        """Returns the a-cycles on the Riemann surface.

        Returns
        -------
        list, RiemannSurfacePath
            A list of the a-cycles of the Riemann surface as
            :class:`RiemannSurfacePath` objects.
        """
        cycles = self.skeleton.a_cycles()
        return [self.RiemannSurfacePath_from_cycle(c) for c in cycles]

    def b_cycles(self):
        """Returns the b-cycles on the Riemann surface.

        Returns
        -------
        list, RiemannSurfacePath
            A list of the b-cycles of the Riemann surface as
            :class:`RiemannSurfacePath` objects.
        """
        cycles = self.skeleton.b_cycles()
        return [self.RiemannSurfacePath_from_cycle(c) for c in cycles]

    def c_cycles(self):
        """Returns the c-cycles of the Riemann surface and the linear
        combination matrix defining the a- and b-cycles from the c-cycles.

        The a- and b- cycles of the Riemann surface are formed from linear
        combinations of the c-cycles. These linear combinations are obtained
        from the :py::meth:`linear_combinations` method.

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
        cycles, linear_combinations = self.skeleton.c_cycles()
        ncols = linear_combinations.shape[1]
        indices = [j for j in range(ncols)
                   if not (linear_combinations[:,j] == 0).all()]
        c_cycles = [cycles[i] for i in indices]
        linear_combinations = linear_combinations[:,indices]

        paths = [self.RiemannSurfacePath_from_cycle(c) for c in c_cycles]
        return paths, linear_combinations

    def RiemannSurfacePath_from_cycle(self, cycle):
        """Constructs a :class:`RiemannSurfacePath` object from a list of
        x-path data.

        Parameters
        ----------
        ypath : list
            A list fo tuples defining a y-path. The output of
            `self.a_cycles()`, `self.b_cycles()`, etc.
        """
        segments = []
        for bi, nrots in cycle:
            gammax_i = self.complex_path_factory.monodromy_path(bi, nrots=nrots)
            segments.extend(gammax_i.segments)

        gammax = ComplexPath(segments)
        gamma = self.RiemannSurfacePath_from_complex_path(gammax)
        return gamma

    def RiemannSurfacePath_from_complex_path(self, complex_path, x0=None, y0=None):
        r"""Constructs a :class:`RiemannSurfacePath` object from x-path data.

        Parameters
        ----------
        complex_path : ComplexPath
            A complex path.
        x0 : complex (default `self.base_point`)
            The starting x-point of the path.
        y0 : complex list (default `self.base_sheets`)
            The starting ordering of the y-sheets.

        Returns
        -------
        RiemannSurfacePath
            A path on the Riemann surface with the prescribed x-path.

        """
        if x0 is None:
            x0 = self.base_point
        if y0 is None:
            y0 = self.base_sheets

        # coerce and assert that x0,y0 lies on the path and curve
        x0 = complex(x0)
        y0 = array(y0, dtype=complex)
        if abs(x0 - complex_path(0)) > 1e-7:
            raise ValueError('The point %s is not at the start of the '
                             'ComplexPath %s'%(x0, complex_path))
        f = self.riemann_surface.f
        curve_error = [abs(complex(f(x0,y0k))) for y0k in y0]
        if max(curve_error) > 1e-7:
            raise ValueError('The fibre %s above %s does not lie on the '
                             'curve %s'%(y0.tolist(), x0, f))

        # build a list of path segments from each tuple of xdata. build a line
        # segment or arc depending on xdata input
        y0_segment = y0
        segments = []
        for segment_x in complex_path.segments:
            # for each segment determine if we're far enough away to use Smale
            # alpha theory paths or if we have to use Puiseux. we add a
            # relazation factor to the radius to account for monodromy paths
            xend = segment_x(1.0)
            b = self.complex_path_factory.closest_discriminant_point(xend)
            R = self.complex_path_factory.radius(b)
            if abs(xend - b) > 0.9*R:
                segment = RiemannSurfacePathSmale(
                    self.riemann_surface, segment_x, y0_segment)
            else:
                segment = RiemannSurfacePathPuiseux(
                    self.riemann_surface, segment_x, y0_segment)

            # determine the starting place of the next segment
            y0_segment = segment.get_y(1.0)
            segments.append(segment)

        # build the entire path from the path segments
        return RiemannSurfacePath(self.riemann_surface, complex_path, y0,
                                  segments)
