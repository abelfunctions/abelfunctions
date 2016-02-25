import unittest

from abelfunctions.tests.test_abelfunctions import AbelfunctionsTestCase
from abelfunctions.differentials import (
    mnuk_conditions,
    recenter_curve,
    differentials_numerators,
    differentials
)

from sage.rings.rational_field import QQ

class DummyRS:
    def __init__(self, f):
        self.f = f

class TestDifferentialsNumerators(AbelfunctionsTestCase):

    def test_f1(self):
        x,y = self.f1.parent().gens()
        a = differentials_numerators(self.f1)
        b = []
        self.assertEqual(a,b)

    def test_f2(self):
        x,y = self.f2.parent().gens()
        a = differentials_numerators(self.f2)
        b = [x*y, x**3]
        self.assertEqual(a,b)

    def test_f4(self):
        x,y = self.f4.parent().gens()
        a = differentials_numerators(self.f4)
        b = []
        self.assertEqual(a,b)

    # def test_f5(self):
    #     x,y = self.f5.parent().gens()
    #     a = differentials_numerators(self.f5)
    #     b = [(x**2 + y**2)]
    #     self.assertEqual(a,b)

    def test_f7(self):
        x,y = self.f7.parent().gens()
        a = differentials_numerators(self.f7)
        b = [1, y, x, x**2]
        self.assertEqual(a,b)

    def test_f8(self):
        x,y = self.f8.parent().gens()
        a = differentials_numerators(self.f8)
        b = [y, x*y**3, x*y**4]
        self.assertEqual(a,b)

class TestDifferentials(AbelfunctionsTestCase):

    def test_f1(self):
        x,y = self.f1.parent().gens()
        dfdy = self.f1.derivative(y)
        X = DummyRS(self.f1)
        a = map(lambda omega: omega.as_expression(), differentials(X))
        b = []
        self.assertEqual(a,b)

    def test_f2(self):
        x,y = self.f2.parent().gens()
        dfdy = self.f2.derivative(y)
        X = DummyRS(self.f2)
        a = map(lambda omega: omega.as_expression(), differentials(X))
        b = [x*y/dfdy, x**3/dfdy]
        self.assertEqual(a,b)

