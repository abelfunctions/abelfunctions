import sympy
import unittest

from abelfunctions.singularities import (
    singularities,
    singular_points_finite,
    singular_points_infinite,
    _transform,
    genus,
    )
from .test_abelfunctions import AbelfunctionsTestCase

from sage.rings.rational_field import QQ
from sage.rings.qqbar import QQbar
from sage.rings.infinity import infinity

class TestSmooth(AbelfunctionsTestCase):

    def test_smooth1(self):
        pass

    def test_smooth2(self):
        pass


class TestSingularPointsFinite(AbelfunctionsTestCase):
    def test_f1(self):
        s = singular_points_finite(self.f1)
        s_actual = [(0,0,1)]
        self.assertEqual(s,s_actual)

    def test_f2(self):
        s = singular_points_finite(self.f2)
        s_actual = [(0,0,1)]
        self.assertEqual(s,s_actual)

    def test_f3(self):
        s = singular_points_finite(self.f3)
        s_actual = [(0,0,1),(1,-1,1),(1,1,1)]
        self.assertEqual(s,s_actual)

    def test_f4(self):
        s = singular_points_finite(self.f4)
        s_actual = [(0,0,1)]
        self.assertEqual(s,s_actual)

    # def test_f5(self):
    #     pass

    def test_f6(self):
        s = singular_points_finite(self.f6)
        s_actual = [(0,0,1)]
        self.assertEqual(s,s_actual)

    # the remaining curves only have singular points at infinity
    def test_f7(self):
        s = singular_points_finite(self.f7)
        s_actual = []
        self.assertEqual(s,s_actual)

    def test_f8(self):
        s = singular_points_finite(self.f8)
        s_actual = []
        self.assertEqual(s,s_actual)

    def test_f9(self):
        s = singular_points_finite(self.f9)
        s_actual = []
        self.assertEqual(s,s_actual)

    def test_f10(self):
        s = singular_points_finite(self.f10)
        s_actual = []
        self.assertEqual(s,s_actual)

    def test_issue71(self):
        R = QQ['x,y']
        x,y = R.gens()
        f = -x**5 + x + y**3
        s = singular_points_finite(f)
        s_actual = []
        self.assertEqual(s,s_actual)


class TestSingularPointsInfinite(AbelfunctionsTestCase):
    def test_f1(self):
        s = singular_points_infinite(self.f1)
        s_actual = [(0,1,0)]
        self.assertEqual(s,s_actual)

    def test_f2(self):
        s = singular_points_infinite(self.f2)
        s_actual = [(0,1,0)]
        self.assertEqual(s,s_actual)

    def test_f3(self):
        s = singular_points_infinite(self.f3)
        s_actual = []
        self.assertEqual(s,s_actual)

    def test_f4(self):
        s = singular_points_infinite(self.f4)
        s_actual = []
        self.assertEqual(s,s_actual)

    # def test_f5(self):
    #     pass

    def test_f6(self):
        s = singular_points_infinite(self.f6)
        s_actual = [(1,0,0)]
        self.assertEqual(s,s_actual)

    # the remaining curves only have singular points at infinity
    def test_f7(self):
        s = singular_points_infinite(self.f7)
        s_actual = [(0,1,0)]
        self.assertEqual(s,s_actual)

    def test_f8(self):
        s = singular_points_infinite(self.f8)
        s_actual = [(1,0,0),(0,1,0)]
        self.assertEqual(s,s_actual)

    def test_f9(self):
        s = singular_points_infinite(self.f9)
        s_actual = [(0,1,0)]
        self.assertEqual(s,s_actual)

    def test_f10(self):
        s = singular_points_infinite(self.f10)
        s_actual = [(1,0,0),(0,1,0)]
        self.assertEqual(s,s_actual)

    def test_issue71(self):
        R = QQ['x,y']
        x,y = R.gens()
        f = -x**5 + x + y**3
        s = singular_points_infinite(f)
        s_actual = [(0,1,0)]
        self.assertEqual(s,s_actual)


class TestTransform(AbelfunctionsTestCase):

    def test_gamma0(self):
        # no-op
        s = (0,0,1)
        f,alpha,beta = _transform(self.f1, s)
        self.assertEqual(self.f1, f)

        f,alpha,beta = _transform(self.f2, s)
        self.assertEqual(self.f2, f)

        f,alpha,beta = _transform(self.f3, s)
        self.assertEqual(self.f3, f)

    def test_alpha0(self):
        s = (0,1,0)
        f,_,_ = _transform(self.f1, s)
        u,v = f.parent().gens()
        g = u**2 - u*v + v**2 - 2*u**2*v + u**4
        self.assertEqual(f, g)

    def test_beta0(self):
        s = (1,0,0)
        f,_,_ = _transform(self.f1, s)
        u,v = f.parent().gens()
        g = u**2 - u**2*v + u**2*v**2 - 2*u*v + 1
        self.assertEqual(f, g)


class TestSingularities(AbelfunctionsTestCase):

    def test_f1(self):
        s = singularities(self.f1)
        s_actual = [
            ((0,0,1),(2,2,1)),
            ((0,1,0),(2,1,2)),
        ]
        self.assertEqual(s,s_actual)

    def test_f2(self):
        s = singularities(self.f2)
        s_actual = [
            ((0,0,1),(3,4,2)),
            ((0,1,0),(4,9,1)),
        ]
        self.assertEqual(s,s_actual)

    def test_f3(self):
        s = singularities(self.f3)
        s_actual = [
            ((0,0,1),(2,1,2)),
            ((1,-1,1),(2,1,2)),
            ((1,1,1),(2,1,2)),
        ]
        self.assertEqual(s,s_actual)

    def test_f4(self):
        s = singularities(self.f4)
        s_actual = [((0,0,1),(2,1,2))]
        self.assertEqual(s,s_actual)

    def test_f5(self):
        pass

    def test_f6(self):
        s = singularities(self.f6)
        s_actual = [
            ((0,0,1),(2,2,2)),
            ((1,0,0),(2,2,2)),
        ]
        self.assertEqual(s,s_actual)

    def test_f7(self):
        s = singularities(self.f7)
        s_actual = [((0,1,0),(3,6,3))]
        self.assertEqual(s,s_actual)

    def test_f8(self):
        s = singularities(self.f8)
        s_actual = [
            ((1,0,0),(5,14,1)),
            ((0,1,0),(2,4,2)),
        ]
        self.assertEqual(s,s_actual)

    def test_f9(self):
        s = singularities(self.f9)
        s_actual = [((0,1,0),(5,12,1))]
        self.assertEqual(s,s_actual)

    def test_f10(self):
        s = singularities(self.f10)
        s_actual = [
            ((1,0,0),(4,6,4)),
            ((0,1,0),(3,6,1)),
        ]
        self.assertEqual(s,s_actual)

    def test_issue71(self):
        R = QQ['x,y']
        x,y = R.gens()
        f = -x**5 + x + y**3
        s = singularities(f)
        s_actual = [((0,1,0),(2,2,1))]
        self.assertEqual(s,s_actual)


class TestGenus(AbelfunctionsTestCase):

    def test_f1(self):
        g = genus(self.f1)
        self.assertEqual(g, 0)

    def test_f2(self):
        g = genus(self.f2)
        self.assertEqual(g, 2)

    def test_f3(self):
        g = genus(self.f3)
        self.assertEqual(g, 0)

    def test_f4(self):
        g = genus(self.f4)
        self.assertEqual(g, 0)

    def test_f5(self):
        g = genus(self.f5)
        self.assertEqual(g, 1)

    def test_f6(self):
        # curve is reducible
        g = genus(self.f6)
        self.assertEqual(g, -1)

    def test_f7(self):
        g = genus(self.f7)
        self.assertEqual(g, 4)

    def test_f8(self):
        g = genus(self.f8)
        self.assertEqual(g, 3)

    def test_f9(self):
        g = genus(self.f9)
        self.assertEqual(g, 9)

    def test_f10(self):
        g = genus(self.f10)
        self.assertEqual(g, 3)

    def test_issue71(self):
        R = QQ['x,y']
        x,y = R.gens()
        f = -x**5 + x + y**3
        g = genus(f)
        self.assertEqual(g, 4)
