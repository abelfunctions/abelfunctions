#from .test_abelfunctions import AbelfunctionsTestCase
import unittest
from abelfunctions.puiseux import puiseux
from abelfunctions.riemann_surface import RiemannSurface
from abelfunctions.integralbasis import integral_basis
from abelfunctions.singularities import singularities, genus
from abelfunctions.differentials import differentials
from .test_abelfunctions import AbelfunctionsTestCase

import sympy
from sympy.abc import x,y,z,t
from sympy import Poly, Point, Segment, Polygon, RootOf, sqrt, S, I

_z = sympy.Symbol('_z')

class TestRCVPaper(unittest.TestCase):

    def setUp(self):
        self.f = x**2*y**3 - x**4 + 1
        self.X = RiemannSurface(self.f,x,y)

    def test_puiseux_0(self):
        # test over x=0
        places = puiseux(self.f,x,y,0,order=32)
        P = places[0]
        self.assertEqual(len(places), 1)

        xpart = -t**3
        ypart = -(y + 1)/t**2
        self.assertEqual(P.xpart, xpart)
        self.assertEqual(P.ypart.expand(), ypart.expand())

        P.extend(order=22)
        extension_polynomial = Poly(-t**24/9 - t**12/3, y)
        self.assertEqual(P._g, extension_polynomial)

    def test_puiseux_1(self):
        # test over x=1
        places = puiseux(self.f,x,y,1,order=7)
        P = places[0]
        self.assertEqual(len(places), 1)

        xpart = t**3/4 + 1
        ypart = t*(y + 1)
        self.assertEqual(P.xpart, xpart)
        self.assertEqual(P.ypart, ypart)

        P.extend(order=10)
        extension_polynomial =  Poly(11*t**6/576 - t**3/24, y)
        self.assertEqual(P._g, extension_polynomial)

    def test_puiseux_I(self):
        # test over x=I
        places = puiseux(self.f,x,y,I,order=7)
        P = places[0]
        self.assertEqual(len(places), 1)

        xpart = -I*t**3/4 + I
        ypart = t*(y + 1)
        self.assertEqual(P.xpart, xpart)
        self.assertEqual(P.ypart, ypart)

        P.extend(order=10)
        extension_polynomial = Poly(11*t**6/576 + t**3/24, y)
        self.assertEqual(P._g, extension_polynomial)

    def test_puiseux_oo(self):
        # test over x=oo
        places = puiseux(self.f,x,y,sympy.oo,order=32)
        P = places[0]
        self.assertEqual(len(places), 1)

        xpart = t**(-3)
        ypart = (y + 1)/t**2
        self.assertEqual(P.xpart, xpart)
        self.assertEqual(P.ypart, ypart)

        P.extend(order=22)
        extension_polynomial = Poly(-t**24/9 - t**12/3, y)
        self.assertEqual(P._g, extension_polynomial)

    def test_puiseux_2(self):
        # test over x=2
        places = puiseux(self.f,x,y,2,order=2)
        P = places[0]
        self.assertEqual(len(places), 3)

        _y = sympy.Symbol('_y')
        r0 = RootOf(4*_y**3-15,0,radicals=False)
        xpart = t + 2
        ypart = y + r0
        self.assertEqual(P.xpart, xpart)
        self.assertEqual(P.ypart, ypart)

        P.add_term()
        extension_polynomial = Poly(17*t*r0/45, y)
        self.assertEqual(P._g, extension_polynomial)

    def test_integral_basis(self):
        actual = [1, y*x, y**2*x**2]
        self.assertItemsEqual(integral_basis(self.f,x,y),
                              actual)

    def test_singularities(self):
        actual = [((0, 1, 0), (2, 2, 1))]
        self.assertItemsEqual(singularities(self.f,x,y),
                              actual)

    def test_genus(self):
        g = genus(self.f,x,y)
        self.assertEqual(g,4)

    def test_differentials(self):
        basis = [1/(3*y**2*x**2), 1/(3*x*y**2), 1/(3*y*x), 1/(3*y**2)]
        self.assertItemsEqual(map(lambda omega: omega.as_sympy_expr(),
                                  self.X.holomorphic_differentials()),
                              basis)
