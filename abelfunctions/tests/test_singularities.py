import sympy
import unittest

from abelfunctions.singularities import (
    singularities, singular_points_finite, singular_points_infinite
    )
from sympy import sympify, RootOf
from .test_abelfunctions import AbelfunctionsTestCase


class TestSmooth(AbelfunctionsTestCase):

    def test_smooth1(self):
        pass

    def test_smooth2(self):
        pass


class TestSingularPointsFinite(AbelfunctionsTestCase):
    def test_f1(self):
        s = singular_points_finite(self.f1,self.x,self.y)
        s_actual = sympify([(0,0,1)])
        self.assertEqual(s,s_actual)

    def test_f2(self):
        s = singular_points_finite(self.f2,self.x,self.y)
        s_actual = sympify([(0,0,1)])
        self.assertEqual(s,s_actual)

    def test_f3(self):
        s = singular_points_finite(self.f3,self.x,self.y)
        s_actual = sympify([(0,0,1),(1,-1,1),(1,1,1)])
        self.assertItemsEqual(s,s_actual)

    def test_f4(self):
        s = singular_points_finite(self.f4,self.x,self.y)
        s_actual = sympify([(0,0,1)])
        self.assertEqual(s,s_actual)

    def test_f5(self):
        pass

    def test_f6(self):
        s = singular_points_finite(self.f6,self.x,self.y)
        s_actual = sympify([(0,0,1)])
        self.assertEqual(s,s_actual)

    # the remaining curves only have singular points at infinity
    def test_f7(self):
        s = singular_points_finite(self.f7,self.x,self.y)
        s_actual = []
        self.assertEqual(s,s_actual)

    def test_f8(self):
        s = singular_points_finite(self.f8,self.x,self.y)
        s_actual = []
        self.assertEqual(s,s_actual)

    def test_f9(self):
        s = singular_points_finite(self.f9,self.x,self.y)
        s_actual = []
        self.assertEqual(s,s_actual)

    def test_f10(self):
        s = singular_points_finite(self.f10,self.x,self.y)
        s_actual = []
        self.assertEqual(s,s_actual)

    def test_issue71(self):
        x = self.x
        y = self.y
        f = -x**5 + x + y**3
        s = singular_points_finite(f,x,y)
        s_actual = []
        self.assertEqual(s,s_actual)


class TestSingularPointsInfinite(AbelfunctionsTestCase):
    def test_f1(self):
        z = sympy.Symbol('z')
        s = singular_points_infinite(self.f1,self.x,self.y,z)
        s_actual = sympify([(0,1,0)])
        self.assertEqual(s,s_actual)

    def test_f2(self):
        z = sympy.Symbol('z')
        s = singular_points_infinite(self.f2,self.x,self.y,z)
        s_actual = sympify([(0,1,0)])
        self.assertEqual(s,s_actual)

    def test_f3(self):
        z = sympy.Symbol('z')
        s = singular_points_infinite(self.f3,self.x,self.y,z)
        s_actual = []
        self.assertEqual(s,s_actual)

    def test_f4(self):
        z = sympy.Symbol('z')
        s = singular_points_infinite(self.f4,self.x,self.y,z)
        s_actual = []
        self.assertEqual(s,s_actual)

    def test_f5(self):
        pass

    def test_f6(self):
        z = sympy.Symbol('z')
        s = singular_points_infinite(self.f6,self.x,self.y,z)
        s_actual = sympify([(1,0,0)])
        self.assertEqual(s,s_actual)

    # the remaining curves only have singular points at infinity
    def test_f7(self):
        z = sympy.Symbol('z')
        s = singular_points_infinite(self.f7,self.x,self.y,z)
        s_actual = sympify([(0,1,0)])
        self.assertEqual(s,s_actual)

    def test_f8(self):
        z = sympy.Symbol('z')
        s = singular_points_infinite(self.f8,self.x,self.y,z)
        s_actual = sympify([(0,1,0),(1,0,0)])
        self.assertItemsEqual(s,s_actual)

    def test_f9(self):
        z = sympy.Symbol('z')
        s = singular_points_infinite(self.f9,self.x,self.y,z)
        s_actual = sympify([(0,1,0)])
        self.assertEqual(s,s_actual)

    def test_f10(self):
        z = sympy.Symbol('z')
        s = singular_points_infinite(self.f10,self.x,self.y,z)
        s_actual = sympify([(0,1,0),(1,0,0)])
        self.assertItemsEqual(s,s_actual)

    def test_issue71(self):
        x = self.x
        y = self.y
        z = sympy.Symbol('z')
        f = -x**5 + x + y**3
        s = singular_points_infinite(f,x,y,z)
        s_actual = sympify([(0,1,0)])
        self.assertEqual(s,s_actual)


class TestSingularities(AbelfunctionsTestCase):

    def test_f1(self):
        s = singularities(self.f1,self.x,self.y)
        s = sympify(s)
        s_actual = sympify(
            [((0,0,1),(2,2,1)),
             ((0,1,0),(2,1,2))]
        )
        self.assertItemsEqual(s,s_actual)

    def test_f2(self):
        s = singularities(self.f2,self.x,self.y)
        s = sympify(s)
        s_actual = sympify(
            [((0,0,1),(3,4,2)),
             ((0,1,0),(4,9,1))]
        )
        self.assertItemsEqual(s,s_actual)

    def test_f3(self):
        s = singularities(self.f3,self.x,self.y)
        s = sympify(s)
        s_actual = sympify(
            [((0,0,1),(2,1,2)),
             ((1,-1,1),(2,1,2)),
             ((1,1,1),(2,1,2))]
        )
        self.assertItemsEqual(s,s_actual)

    def test_f4(self):
        s = singularities(self.f4,self.x,self.y)
        s = sympify(s)
        s_actual = sympify(
            [((0,0,1),(2,1,2))]
        )
        self.assertItemsEqual(s,s_actual)

    def test_f5(self):
        pass

    def test_f6(self):
        s = singularities(self.f6,self.x,self.y)
        s = sympify(s)
        s_actual = sympify(
            [((0,0,1),(2,2,2)),
             ((1,0,0),(2,2,2))]
        )
        self.assertItemsEqual(s,s_actual)

    def test_f7(self):
        s = singularities(self.f7,self.x,self.y)
        s = sympify(s)
        s_actual = sympify(
            [((0,1,0),(3,6,3))]
        )
        self.assertItemsEqual(s,s_actual)

    def test_f8(self):
        s = singularities(self.f8,self.x,self.y)
        s = sympify(s)
        s_actual = sympify(
            [((0,1,0),(2,4,2)),
             ((1,0,0),(5,14,1))]
        )
        self.assertItemsEqual(s,s_actual)

    def test_f9(self):
        s = singularities(self.f9,self.x,self.y)
        s = sympify(s)
        s_actual = sympify(
            [((0,1,0),(5,12,1))]
        )
        self.assertItemsEqual(s,s_actual)

    def test_f10(self):
        s = singularities(self.f10,self.x,self.y)
        s = sympify(s)
        s_actual = sympify(
            [((0,1,0),(3,6,1)),
             ((1,0,0),(4,6,4))]
        )
        self.assertItemsEqual(s,s_actual)

    def test_issue71(self):
        x = self.x
        y = self.y
        f = -x**5 + x + y**3
        s = singularities(f,x,y)
        s_actual = sympify(
            [((0,1,0),(2,2,1))]
        )
        self.assertEqual(s,s_actual)
