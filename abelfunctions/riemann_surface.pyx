"""
RiemannSurfaces
===============

Authors
-------

* Chris Swierczewski (January 2014)
"""

import sympy

from .riemann_surface_path_factory import RiemannSurfacePathFactory
from .differentials import differentials
from .singularities import genus


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

    def __init__(self, f, x, y, base_point=None, base_sheets=None, kappa=3./5):
        """Construct a Riemann surface.

        Arguments
        ---------
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
        self.PathFactory = RiemannSurfacePathFactory(self)

    def __repr__(self):
        s = 'Riemann surface defined by the algebraic curve %s'%(self.f)
        return s

    def __call__(self, alpha, beta):
        pass

    def show_paths(self):
        self.PathFactory.XSkel.show_paths()

    def point(self, alpha, beta):
        raise NotImplementedError('Need to define a RiemannSurfacePoint '
                                  'class before this can work as intended.')

    #
    # Monodromy: expose some methods / properties of self.Monodromy
    # without subclassing (since it doesn't make sense that a Riemann
    # surface is a type of Monodromy group.)
    #
    def monodromy_group(self):
        return self.PathFactory.monodromy_group()

    def base_point(self):
        return self.PathFactory.base_point()

    def base_sheets(self):
        return self.PathFactory.base_sheets()

    def base_lift(self):
        return self.base_sheets()

    def branch_points(self):
        return self.PathFactory.branch_points()

    def holomorphic_differentials(self):
        """Returns a basis of holomorphic differentials defined on the Riemann
        surface.

        """
        return differentials(self.f, self.x, self.y)

    def differentials(self):
        return self.holomorphic_differentials()

    def genus(self):
        return genus(self.f, self.x, self.y)



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
