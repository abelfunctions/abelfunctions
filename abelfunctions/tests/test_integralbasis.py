import unittest

from abelfunctions.integralbasis import integral_basis
from abelfunctions.tests.test_abelfunctions import AbelfunctionsTestCase

from sage.all import SR
from sage.rings.big_oh import O
from sage.rings.laurent_series_ring import LaurentSeriesRing
from sage.rings.rational_field import QQ
from sage.rings.qqbar import QQbar

class TestIntegralBasis(AbelfunctionsTestCase):

    def test_f1(self):
        x,y = self.f1.parent().gens()
        a = integral_basis(self.f1)
        b = [1, y*(x**2-x+1)/x**2]
        self.assertEqual(a,b)

    def test_f2(self):
        x,y = self.f2.parent().gens()
        a = integral_basis(self.f2)
        b = [1, y/x, y**2/x**3]
        self.assertEqual(a,b)

    def test_f3(self):
        x,y = self.f3.parent().gens()
        a = integral_basis(self.f3)
        b = [1, y, (y**2-1)/(x-1), -y*(x - 4*y**2 + 3)/(x*(x - 1))]
        self.assertEqual(a,b)

    def test_f4(self):
        x,y = self.f4.parent().gens()
        a = integral_basis(self.f4)
        b = [1, y/x]
        self.assertEqual(a,b)

    def test_f5(self):
        x,y = self.f5.parent().gens()
        a = integral_basis(self.f5)
        b = [1, y, y**2, y**3, y*(y**3-1)/x, y**2*(y**3-1)/x**2]
        self.assertEqual(a,b)

    def test_f6(self):
        x,y = self.f6.parent().gens()
        a = integral_basis(self.f6)
        b = [1, y, y**2/x, y**3/x]
        self.assertEqual(a,b)

    def test_f7(self):
        x,y = self.f7.parent().gens()
        a = integral_basis(self.f7)
        b = [1, y, y**2]
        self.assertEqual(a,b)

    def test_f8(self):
        x,y = self.f8.parent().gens()
        a = integral_basis(self.f8)
        b = [1, x*y, x*y**2, y**3*x, x**2*y**4, y**5*x**2]
        self.assertEqual(a,b)

    def test_f8a(self):
        # the curve f8 recentered at the singular point (1 : 0 : 0)
        R = QQ['x,y']
        x,y = R.gens()
        g = -y**8 + x**6 + 2*x**5
        a = integral_basis(g)
        b = [1, y, y**2/x, y**3/x, y**4/x**2, y**5/x**3, y**6/x**3, y**7/x**4]
        self.assertEqual(a,b)

    def test_f8b(self):
        # the curve f8 recentered at the singular point (0 : 1 : 0)
        R = QQ['x,y']
        x,y = R.gens()
        g = -y**8 + 2*x**3 + x**2
        a = integral_basis(g)
        b = [1, y, y**2, y**3, y**4/x, y**5/x, y**6/x, y**7/x]
        self.assertEqual(a,b)

    def test_f9(self):
        x,y = self.f9.parent().gens()
        a = integral_basis(self.f9)
        b = [1, y, y**2]
        self.assertEqual(a,b)

    def test_f10(self):
        x,y = self.f10.parent().gens()
        a = integral_basis(self.f10)
        b = [1, x*y, x**2*y**2, x**3*y**3]
        self.assertEqual(a,b)

    def test_rcvexample(self):
        R = QQ['x,y']
        x,y = R.gens()
        f = x**2*y**3 - x**4 + 1
        a = integral_basis(f)
        b = [1, x*y, x**2*y**2]
        self.assertEqual(a,b)

    def test_rcvexample_monicized(self):
        R = QQ['x,y']
        x,y = R.gens()
        f = y**3 - x**8 + x**4
        a = integral_basis(f)
        b = [1, y/x, y**2/x**2]
        self.assertEqual(a,b)

    def test_issue86(self):
        R = QQ['x,z']
        x,z = R.gens()
        f = -x**7 + 2*x*z**5 + z**4
        a = integral_basis(f)
        b = [1, 2*x*z, 2*z*(2*x*z + 1)/x, 4*z**2*(2*x*z + 1)/x**3,
             8*z**3*(2*x*z + 1)/x**5]
        self.assertEqual(a,b)
