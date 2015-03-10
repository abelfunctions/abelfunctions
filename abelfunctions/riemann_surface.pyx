"""
RiemannSurfaces
===============

Authors
-------

* Chris Swierczewski (January 2014)
"""

import numpy
import sympy
import scipy
import scipy.integrate
import scipy.linalg

from .differentials import differentials
from .differentials import Differential
from .differentials cimport Differential
from .divisor import Place, DiscriminantPlace, RegularPlace, Divisor
from .puiseux import puiseux
from .riemann_surface_path import RiemannSurfacePathPrimitive
from .riemann_surface_path cimport RiemannSurfacePathPrimitive
from .riemann_surface_path_factory import RiemannSurfacePathFactory
from .singularities import genus
from .utilities import rootofsimp


cdef class RiemannSurface:
    """A Riemann surface defined by a complex plane algebraic curve.

    Attributes
    ----------
    f : sympy.Expression
        The algebraic curve representing the Riemann surface.
    x,y : sympy.Symbol
        The dependent and independent variables, respectively.
    """
    property f:
        def __get__(self):
            return self._f
    property x:
        def __get__(self):
            return self._x
    property y:
        def __get__(self):
            return self._y
    property deg:
        def __get__(self):
            return self._deg
    property PF:
        def __get__(self):
            return self.PathFactory

    def __init__(self, f, x, y, base_point=None, base_sheets=None, kappa=2./5):
        """Construct a Riemann surface.

        Parameters
        ----------
        f : sympy.Expression
            The algebraic curve representing the Riemann surface.
        x,y : sympy.Symbol
            The dependent and independent variables, respectively.
        base_point : complex, optional
            A custom base point for the Monodromy group.
        base_sheets : complex list, optional
            A custom ordering of the sheets at the base point.
        kappa : double
            A scaling parameter greater than 0 but less than 1 used to
            define the radii of the x-path circles around the curve's
            branch points.
        """
        self._f = f
        self._x = x
        self._y = y
        self._deg = sympy.degree(f,y)

        # set custom base point, if provided. otherwise, base_point is
        # set by self.discriminant_points()
        self._base_point = base_point
        self._discriminant_points = None
        self._discriminant_points_exact = None
        self.discriminant_points()  # sets the base point of the surface

        # se the base sheets
        if base_sheets:
            self._base_sheets = base_sheets
        else:
            self._base_sheets = self.base_sheets()

        # cache for key calculations
        self._period_matrix = None
        self._riemann_matrix = None
        self._genus = None
        self._holomorphic_differentials = None
        self.PathFactory = RiemannSurfacePathFactory(self)

    def __repr__(self):
        s = 'Riemann surface defined by the algebraic curve %s'%(self.f)
        return s

    def __call__(self, alpha, beta=None):
        r"""Returns a place or places on the Riemann surface.

        Parameters
        ----------
        alpha : complex or sympy.Expr
            The x-projection of the place.
        beta : complex or sympy.Expr (optional)
            If provided, will only return places with the given
            y-projection. There may be multiple places on the surface
            with the same x- and y-projections.

        Returns
        -------
        Place or list of Places
            If multiple places

        """
        # alpha = infinity case
        infinities = ['oo', sympy.oo, numpy.Inf]
        if alpha in infinities:
            p = puiseux(self.f, self.x, self.y, sympy.oo,
                        parametric=True, exact=True)
            return [DiscriminantPlace(self, pi) for pi in p]

        # first coerce b into an exact discriminant point if it's
        # epsilon close to one
        exact = isinstance(alpha,sympy.Expr) or isinstance(alpha,int)
        b = self.closest_discriminant_point(alpha,exact=exact)
        if abs(numpy.complex(alpha) - numpy.complex(b)) < 1e-12:
            # return discriminant places corresponding to x=alpha y=beta
            p = puiseux(self.f,self.x,self.y,b,beta=beta,
                        parametric=True,exact=exact)
            return [DiscriminantPlace(self, pi) for pi in p]

        # return a regular place if far enough away from a
        # discriminant point
        if not beta is None:
            # sanity check that the place is on the curve
            curve_eval = self.f.evalf(subs={self.x:alpha, self.y:beta})
            if abs(curve_eval) > 1e-8:
                raise ValueError('The place (%s, %s) does not lie on '
                                 'the curve / surface.')
            return RegularPlace(self, alpha, beta)

        # otherwise, compute the roots above x=alpha and return a list
        # of places
        _y = sympy.Symbol('_'+str(self.y))
        falpha = self.f.subs({self.x:alpha,self.y:_y}).as_poly(_y)
        yroots = falpha.all_roots(radicals=False)
        return [RegularPlace(self,alpha,beta) for beta in yroots]

    def show_paths(self, ax=None, *args, **kwds):
        """Plots all of the monodromy paths of the curve.

        Arguments
        ---------
        ax : matplotlib.Axes
            The figure axes on which to plot the paths.

        Returns
        -------
        None
        """
        self.PathFactory.show_paths(ax=ax, *args, **kwds)

    def discriminant_points(self, exact=True):
        r"""Returns the discriminant points of the underlying curve.

        A discriminant point :math:`x=b` is an x-point where at least
        one y-root lying above has multiplicity greater than one. A
        :class:`PuiseuxTSeries` is required to represent a place on the
        Riemann surface whose x-projection is a discriminant
        point. These kinds of places are of type
        :class:`DiscriminantPlace`.

        .. note::

            The ordering of the discriminant points is important for the
            purposes of computing the monodromy group, which is done in
            the :class:`RiemannSurfacePathFactory` attribute,
            `PathFactory`.

        Parameters
        ----------
        exact : boolean
            If `True`, returns symbolic discriminant points. Otherwise,
            returns a numerical approximation. Both are cached for
            performance.

        Returns
        -------
        list
            A list of the discriminant points of the underlying curve.

        """
        # use cached discriminant points if available
        if not self._discriminant_points is None:
            if exact:
                return self._discriminant_points_exact
            return self._discriminant_points

        # compute the symbolic and numerical discriminant points
        f = self.f
        x = self.x
        y = self.y
        p = sympy.Poly(f,[x,y])
        _x = sympy.Symbol('_'+str(x))
        res = sympy.resultant(p,p.diff(y),y).subs(x,_x).as_poly(_x)
        rts = res.all_roots(multiple=False, radicals=False)
        rts, multiplicities = zip(*rts)
        discriminant_points_exact = numpy.array(rts)
        discriminant_points = discriminant_points_exact.astype(numpy.complex)

        # determine a base_point, if not specified
        if self._base_point is None:
            a = min(bi.real - 1 for bi in discriminant_points)
            self._base_point = a

        # sort the discriminant points first by argument with the base
        # point and then by distance from the base point.
        centered_points = discriminant_points - a
        distances = numpy.abs(centered_points)
        arguments = numpy.angle(centered_points)
        sort_index = numpy.lexsort((distances, arguments))

        # cache and return
        self._discriminant_points_exact = discriminant_points_exact[sort_index]
        self._discriminant_points = discriminant_points[sort_index]
        if exact:
            return self._discriminant_points_exact
        return self._discriminant_points

    def closest_discriminant_point(self, x, exact=True):
        r"""Returns the closest discriminant point to a point x.

        An often-used helper function by several components of
        :class:`RiemannSurface`.

        Parameters
        ----------
        x : complex
            A complex x-point.
        exact : boolean
            If `True`, returns a `sympy.Expr` representing the
            discriminant point exactly. Otherwise, returns a numerical
            approximation.

        Returns
        -------
        complex or sympy.Expr
            The discriminant point, either exact or numerical.
        """
        b = self.discriminant_points(exact=exact)
        bf = self.discriminant_points(exact=False)

        # for performance, coerce everything to floating point
        # approximations. if the discriminant points are less than 1e-16
        # apart then we're screwed, anyway.
        x = numpy.complex(x)
        idx = numpy.argmin(numpy.abs(bf - x))
        return b[idx]

    # Monodromy: expose some methods / properties of self.Monodromy
    # without subclassing (since it doesn't make sense that a Riemann
    # surface is a type of Monodromy group.)
    def monodromy_group(self):
        return self.PathFactory.monodromy_group()

    def base_point(self):
        r"""Returns the base x-point of the Riemann surface.
        """
        return self._base_point

    def base_place(self):
        r"""Returns the base place of the Riemann surface.

        The base place is the place from which all paths on the Riemann
        surface are constructed. The AbelMap begins integrating from the
        base place.

        Parameters
        ----------
        None

        Returns
        -------
        Place

        """
        places = self(self._base_point)
        return places[0]

    def base_sheets(self):
        r"""Returns the base sheets of the Riemann surface.

        The base sheets are the y-roots lying above the base point of
        the surface.  The base place of the Riemann surface is given by
        the base x-point and the first element of the base sheets.

        Parameters
        ----------
        None

        Returns
        -------
        list, complex
            An ordered list of roots lying above the base point of the
            curve.

        """
        # returned cached base sheets if availabe
        if not self._base_sheets is None:
            return self._base_sheets
        self._base_sheets = self.lift(self._base_point)
        return self._base_sheets

    def lift(self, x):
        r"""List the x-point `x` to the fibre of y-roots.

        Basically, computes the y-roots of :math:`f(x,y) = 0` for the
        given `x`.

        .. note::

            The y-roots are given in no particular order. Be careful
            when using these to construct :class:`RiemannSurfacePath`
            objects.

        Parameters
        ----------
        x : complex

        Returns
        -------
        list, complex
        """
        # compute the base sheets
        p = self._f.as_poly(self._y)
        coeffs = numpy.array(
            [c.evalf(subs={self._x:x}, n=15) for c in p.all_coeffs()],
            dtype=numpy.complex)
        poly = numpy.poly1d(coeffs)
        lift = poly.r
        return lift

    def base_lift(self):
        r"""Same as :meth:`base_sheets`."""
        return self.base_sheets()

    def branch_points(self):
        return self.PathFactory.branch_points()

    def holomorphic_differentials(self):
        r"""Returns a basis of holomorphic differentials on the surface.

        Parameters
        ----------
        None

        Returns
        -------
        list, HolomorphicDifferential

        """
        f,x,y = self.f, self.x, self.y
        if not self._holomorphic_differentials:
            self._holomorphic_differentials = differentials(self)
        return self._holomorphic_differentials

    def holomorphic_oneforms(self):
        r"""Alias for :meth:`holomorphic_differentials`."""
        return self.holomorphic_differentials()

    def genus(self):
        if not self._genus:
            self._genus = genus(self.f,self.x,self.y)
        return self._genus

    def a_cycles(self):
        return self.PF.a_cycles()

    def b_cycles(self):
        return self.PF.b_cycles()

    def c_cycles(self):
        return self.PF.c_cycles()

    def path(self, P, P0=None):
        r"""Constructs a path to the place `P`.

        Parameters
        ----------
        P : Place
            The place

        Returns
        -------
        RiemannSurfacePath

        """
        return self.PathFactory.path_to_place(P)

    cpdef complex integrate(self, Differential omega,
                            RiemannSurfacePathPrimitive gamma):
        r"""Integrate the differential `omega` over the path `gamma`.

        Parameters
        ----------
        omega : Differenial
            A differential defined on the Riemann surface.
        gamma : RiemannSurfacePathPrimitive
            A continuous path on the Riemann surface.

        Returns
        -------
        complex
            The integral of `omega` on `gamma`.

        """
        cdef complex value = gamma.integrate(omega)
        return value

    def period_matrix(self):
        r"""Returns the period matrix of the Riemann surface.

        The period matrix is obtained by integrating a basis of
        holomorphic one-forms over a first homology group basis.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.array
            A :math:`g \times 2g` complex matrix of periods.

        """
        if not (self._period_matrix is None):
            return self._period_matrix

        c_cycles, linear_combinations = self.c_cycles()
        oneforms = self.holomorphic_oneforms()
        c_periods = []
        g = self.genus()
        m = len(c_cycles)

        for omega in oneforms:
            omega_periods = []
            for gamma in c_cycles:
                omega_periods.append(self.integrate(omega, gamma))
            c_periods.append(omega_periods)

        # take appropriate linear combinations of the c-periods to
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

        self._period_matrix = tau
        return self._period_matrix

    def riemann_matrix(self):
        r"""Returns the Riemann matrix of the Riemann surface.

        A Riemann matrix of the surface is obtained by normalizing the
        chosen basis of holomorphic differentials.

        .. math::

            \tau = [A \; B] = [I \; \Omega]

        Parameters
        ----------
        None

        Returns
        -------
        numpy.array
            A :math:`g \times g` Riemann matrix corresponding to the
            Riemann surface.

        """
        if not self._riemann_matrix is None:
            return self._riemann_matrix

        g = self.genus()
        tau = self.period_matrix()
        A = tau[:,:g]
        B = tau[:,g:]
        self._riemann_matrix = numpy.dot(scipy.linalg.inv(A), B)
        return self._riemann_matrix




if __name__ == '__main__':
    import sympy
    from sympy.abc import x,y

    f0 = y**3 - 2*x**3*y - x**8
    f1 = (x**2 - x + 1)*y**2 - 2*x**2*y + x**4
    f2 = -x**7 + 2*x**3*y + y**3
    f3 = (y**2-x**2)*(x-1)*(2*x-3) - 4*(x**2+y**2-2*x)**2
    f4 = y**2 + x**3 - x**2
    f5 = (x**2 + y**2)**3 + 3*x**2*y - y**3
    f6 = y**4 - y**2*x + x**2
    f7 = y**3 - (x**3 + y)**2 + 1
    f8 = (x**6)*y**3 + 2*x**3*y - 1
    f9 = 2*x**7*y + 2*x**7 + y**3 + 3*y**2 + 3*y
    f10 = (x**3)*y**4 + 4*x**2*y**2 + 2*x**3*y - 1
    f11 = y**2 - (x-2)*(x-1)*(x+1)*(x+2)
    f12 = x**4 + y**4 - 1

    f = f2
    X = RiemannSurface(f, x, y)
