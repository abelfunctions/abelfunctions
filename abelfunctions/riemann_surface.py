"""
RiemannSurfaces
===============

Authors
-------

* Chris Swierczewski (January 2014)
"""

import numpy
import scipy
import scipy.linalg

import abelfunctions
from abelfunctions.differentials import differentials as generate_differentials
from abelfunctions.differentials import validate_differentials
from abelfunctions.divisor import Place, DiscriminantPlace, RegularPlace, Divisor
from abelfunctions.puiseux import puiseux
from abelfunctions.riemann_surface_path_factory import RiemannSurfacePathFactory
from abelfunctions.singularities import genus

from sage.all import QQbar, infinity
from sage.misc.cachefunc import cached_function, cached_method


class RiemannSurface(object):
    """A Riemann surface defined by a complex plane algebraic curve :math:`C :
    f(x,y) = 0`.

    """

    @property
    def f(self):
        return self._f

    @property
    def deg(self):
        # legacy
        return self._degree

    @property
    def degree(self):
        return self._degree

    @property
    def base_point(self):
        return self.path_factory.base_point

    @property
    def base_sheets(self):
        return self.path_factory.base_sheets

    @property
    def base_place(self):
        alpha = self.base_point
        beta = self.base_sheets[0]
        return RegularPlace(self, alpha, beta)

    @property
    def branch_points(self):
        return self.path_factory.branch_points

    @property
    def discriminant_points(self):
        return self.path_factory.discriminant_points

    @property
    def path_factory(self):
        if not self._path_factory:
            self._path_factory = RiemannSurfacePathFactory(
                self, base_point=self._base_point,
                base_sheets=self._base_sheets, kappa=self._kappa)
            self._base_point = self._path_factory.base_point
            self._base_sheets = self._path_factory.base_sheets
        return self._path_factory

    @property
    def differentials(self):
        """Provides the differentials defined for this surface.

        Returns user-defined differentials if set, otherwise defaults to
        :meth:`holomorphic_differentials`."""

        if self._user_differentials is None:
            return self.holomorphic_differentials()
        else:
            return self._user_differentials

    @differentials.setter
    def differentials(self, value):
        """Specify a list of Differentials to use or None to clear."""

        if value is not None and not validate_differentials(value, self.genus()):
                raise ValueError('Invalid list of differentials')
        self._user_differentials = value

    def __init__(self, f, base_point=None, base_sheets=None, kappa=3./5):
        """Construct a Riemann surface.

        Parameters
        ----------
        f : curve
            The algebraic curve representing the Riemann surface.
        base_point : complex, optional
            A custom base point for the monodromy group.
        base_sheets : complex list, optional
            A custom ordering of the sheets at the base point.
        kappa : double
            A scaling parameter greater than 0 but less than 1 used to
            define the radii of the x-path circles around the curve's
            branch points.
        """
        R = f.parent()
        x,y = R.gens()
        self._f = f
        self._degree = f.degree(y)
        self._base_point = base_point
        self._base_sheets = base_sheets
        self._kappa = kappa

        # still need to fix base_point / base_sheet interaction
        self._path_factory = None

        self._user_differentials = None

    def __repr__(self):
        s = 'Riemann surface defined by f = %s'%(self.f)
        return s

    def __call__(self, alpha, beta=None):
        r"""Returns a place or places on the Riemann surface with the given x-projection
        and, optionally, given y-projection.

        Parameters
        ----------
        alpha : complex
            The x-projection of the place.
        beta : complex
            If provided, will only return places with the given y-projection.
            There may be multiple places on the surface with the same x- and
            y-projections.

        Returns
        -------
        places : Place or list of Places
            Returns all places on the Riemann surface with x-projection `alpha`
            or x,y-projection `(alpha, beta)`, if `beta` is provided.

        """
        places = self.places(alpha, beta=beta)
        return places

    def places(self, alpha, beta=None):
        r"""Returns a place or places on the Riemann surface with the given x-projection
        and, optionally, given y-projection.

        Parameters
        ----------
        alpha : complex
            The x-projection of the place.
        beta : complex
            If provided, will only return places with the given y-projection.
            There may be multiple places on the surface with the same x- and
            y-projections.

        Returns
        -------
        places : Place or list of Places
            Returns all places on the Riemann surface with x-projection `alpha`
            or x,y-projection `(alpha, beta)`, if `beta` is provided.

        """
        # alpha = infinity case
        infinities = [infinity, 'oo', numpy.Inf]
        if alpha in infinities:
            alpha = infinity
            p = puiseux(self.f, alpha)
            places = [DiscriminantPlace(self, pi) for pi in p]
            return places

        # if alpha is epsilon close to a discriminant point then set it exactly
        # equal to that discriminant point. there is usually no reason to
        # compute a puiseux series so close to a discriminant point
        try:
            alpha = QQbar(alpha)
            exact = True
        except TypeError:
            alpha = numpy.complex(alpha)
            exact = False
        b = self.path_factory.closest_discriminant_point(alpha,exact=exact)

        # if alpha is equal to or close to a discriminant point then return a
        # discriminant place
        if abs(alpha - b) < 1e-12:
            p = puiseux(self.f, b, beta)
            places = [DiscriminantPlace(self, pi) for pi in p]
            return places

        # otherwise, return a regular place if far enough away
        if not beta is None:
            curve_eval = self.f(alpha, beta)
            if abs(curve_eval) > 1e-8:
                raise ValueError('The place (%s, %s) does not lie on the curve '
                                 '/ surface.' % (alpha, beta))
            place = RegularPlace(self, alpha, beta)
            return place

        # if a beta (y-coordinate) is not specified then return all places
        # lying above x=alpha
        R = self.f.parent()
        x,y = R.gens()
        falpha = self.f(alpha,y).univariate_polynomial()
        yroots = falpha.roots(ring=falpha.base_ring(), multiplicities=False)
        places = [RegularPlace(self, alpha, beta) for beta in yroots]
        return places

    def show_paths(self, *args, **kwds):
        """Plots all of the monodromy paths of the curve.

        Parameters
        ----------
        ax : matplotlib.Axes
            The figure axes on which to plot the paths.

        Returns
        -------
        plt : sage plot
        """
        plt = self.path_factory.show_paths(*args, **kwds)
        return plt

    # Monodromy: expose some methods / properties of self.Monodromy without
    # subclassing (since it doesn't make sense that a Riemann surface is a type
    # of Monodromy group.)
    def monodromy_group(self):
        r"""Returns the monodromy group of the underlying curve.

        The monodromy group is represented by a list of four items:

        * `base_point` - a point in the complex x-plane where every monodromy
          path begins and ends,
        * `base_sheets` - the y-roots of the curve lying above `base_point`,
        * `branch_points` - the branch points of the curve,
        * `permutations` - the permutations of he base sheets corresponding to
          each branch point.

        Parameters
        ----------
        None

        Returns
        -------
        monodromy : list
            The monodromy group information described above.
        """
        monodromy = self.path_factory.monodromy_group()
        return monodromy

    def holomorphic_differentials(self):
        r"""Returns a basis of holomorphic differentials on the surface.

        Parameters
        ----------
        None

        Returns
        -------
        differentials : list
            A list of holomorphic differentials forming a basis.

        """
        value = generate_differentials(self)
        return value

    @cached_method
    def genus(self):
        r"""The genus of this Riemann surface.

        Returns
        -------
        genus : int
        """
        g = genus(self.f)
        return g

    def a_cycles(self):
        r"""Returns the a-cycles of the Riemann surface.

        See Also
        --------
        :func:`PathFactory.a_cycles`
        """
        return self.path_factory.a_cycles()

    def b_cycles(self):
        r"""Returns the b-cycles of the Riemann surface.

        See Also
        --------
        :func:`PathFactory.b_cycles`
        """
        return self.path_factory.b_cycles()

    def c_cycles(self):
        r"""Returns the c-cycles of the Riemann surface.

        The c-cycles of a Riemann surface are intermediate cycles from which
        the a- and b-cycles are built. See :func:`PathFactory.c_cycles` and
        :func:`YPathFactory.c_cycles` for more information on how these are
        used.

        See Also
        --------
        :func:`PathFactory.c_cycles`
        """
        return self.path_factory.c_cycles()

    def path(self, P):
        r"""Constructs a path to the place `P`.

        Computes a :class:`RiemannSurfacePath` starting at the base place of
        the surface.

        .. note::

            Currently, target places close to, but not equal to, discriminant
            places of the curve are not allowed.

        Parameters
        ----------
        P : Place
            The target place.

        Returns
        -------
        gamma : RiemannSurfacePath
            A path from :math:`P_0` to :math:`P`.

        See Also
        --------
        :func:`PathFactory.path_to_place`

        """
        gamma = self.path_factory.path_to_place(P)
        return gamma

    def integrate(self, omega, gamma):
        r"""Integrate the differential `omega` over the path `gamma`.

        Parameters
        ----------
        omega : Differenial
            A differential defined on the Riemann surface.
        gamma : RiemannSurfacePathPrimitive
            A continuous path on the Riemann surface.

        Returns
        -------
        integral : complex
            The integral of `omega` on `gamma`.

        """
        integral = gamma.integrate(omega)
        return integral

    @cached_method
    def period_matrix(self):
        r"""Returns the period matrix of the Riemann surface.

        The period matrix is obtained by integrating a basis of holomorphic
        one-forms over a first homology group basis.

        Parameters
        ----------
        None

        Returns
        -------
        tau : matrix.
            A :math:`g \times 2g` complex matrix of periods. The first :math:`g
            \times g` block consists of the a-periods. The second block
            contains the b-periods.

        """
        # compute the c-cycle periods
        c_cycles, linear_combinations = self.c_cycles()
        oneforms = self.differentials
        c_periods = []
        g = self.genus()
        m = len(c_cycles)
        for omega in oneforms:
            omega_periods = []
            for gamma in c_cycles:
                omega_periods.append(self.integrate(omega, gamma))
            c_periods.append(omega_periods)

        # take appropriate linear combinations of the c-periods, using the
        # linear combinations information computed by the path factory, to
        # obtain the a- and b-periods
        #
        # tau[i,j] = \int_{a_j} \omega_i,  j < g
        # tau[i,j] = \int_{b_j} \omega_i,  j >= g
        #
        tau = numpy.zeros((g,2*g), dtype=numpy.complex)
        for i in range(g):
            for j in range(2*g):
                tau[i,j] = sum(linear_combinations[j,k] * c_periods[i][k]
                               for k in range(m))
        return tau

    @cached_method
    def riemann_matrix(self):
        r"""Returns the Riemann matrix of the Riemann surface.

        A Riemann matrix of the surface is obtained by normalizing the chosen
        basis of holomorphic differentials.

        .. math::

            \tau = [A \; B] = [I \; \Omega]

        Parameters
        ----------
        None

        Returns
        -------
        omega : mathrix
            The :math:`g \times g` Riemann matrix of the Riemann surface.

        """
        g = self.genus()
        tau = self.period_matrix()
        A = tau[:,:g]
        B = tau[:,g:]
        omega = numpy.dot(scipy.linalg.inv(A), B)
        return omega
