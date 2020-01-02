import unittest

from abelfunctions.differentials import (
    mnuk_conditions,
    recenter_curve,
    differentials_numerators,
    differentials,
    validate_differentials,
    Differential
)
from abelfunctions.riemann_surface import RiemannSurface
from abelfunctions.tests.test_abelfunctions import AbelfunctionsTestCase

from sage.all import QQ, CC

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

    @unittest.skip("Takes too much time")
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
        a = [omega.as_expression() for omega in differentials(X)]
        b = []
        self.assertEqual(a, b)

    def test_f2(self):
        x,y = self.f2.parent().gens()
        dfdy = self.f2.derivative(y)
        X = DummyRS(self.f2)
        a = [omega.as_expression() for omega in differentials(X)]
        b = [x*y/dfdy, x**3/dfdy]
        self.assertEqual(a, b)

    def test_validation_success(self):
        x,y = self.f2.parent().gens()
        dfdy = self.f2.derivative(y)
        X = DummyRS(self.f2)
        
        diffs = [Differential(X, x*y, dfdy), Differential(X, x**3, dfdy)]
        self.assertTrue(validate_differentials(diffs, 2))

    def test_validation_failures(self):
        x,y = self.f2.parent().gens()
        dfdy = self.f2.derivative(y)
        X = DummyRS(self.f2)
        Y = DummyRS(self.f2)
        
        Xdiffs = [Differential(X, x*y, dfdy), Differential(X, x**3, dfdy)]
        Ydiffs = [Differential(Y, x*y, dfdy), Differential(Y, x**3, dfdy)]
        g = len(Xdiffs)

        # Type failure: add a non-Differential value
        self.assertFalse(validate_differentials(Xdiffs[:-1] + [0], g))

        # Surface failure: defined on different Riemann surfaces
        self.assertFalse(validate_differentials(Xdiffs[:-1] + Ydiffs[-1:], g))

        # Genus failure: too few or too many differentials
        self.assertFalse(validate_differentials(Xdiffs, g + 1))
        self.assertFalse(validate_differentials(Xdiffs, g - 1))


class TestCenteredAtRegularPlace(AbelfunctionsTestCase):
    # tests if differentials are correctly evaluated at regular places on the
    # Riemann surface. see Issue #123.
    #
    # test: check if evaluating the differential at the (x,y)-projection of the
    # place is equal (or, nomerically close to) to evaluating the centered
    # differential at t=0

    def test_f1_regular_places(self):
        X = RiemannSurface(self.f1)
        omegas = differentials(X)

        # the places above x=-1 are regular
        places = X(-1)
        for P in places:
            a,b = P.x,P.y
            for omega in omegas:
                omega_P = omega.centered_at_place(P)
                val1 = omega(a,b)
                val2 = omega_P(CC(0))
                self.assertLess(abs(val1-val2), 1e-8)

    def test_f2_regular_places(self):
        X = self.X2
        omegas = differentials(X)

        # the places above x=1 are regular
        places = X(1)
        for P in places:
            a,b = P.x,P.y
            for omega in omegas:
                omega_P = omega.centered_at_place(P)
                val1 = omega(a,b)
                val2 = omega_P(CC(0))
                self.assertLess(abs(val1-val2), 1e-8)

    def test_hyperelliptic_regular_places(self):
        R = QQ['x,y']
        x,y = R.gens()
        X = RiemannSurface(y**2 - (x+1)*(x-1)*(x-2)*(x+2))
        omegas = differentials(X)

        # the places above x=0 are regular
        places = X(0)
        for P in places:
            a,b = P.x,P.y
            for omega in omegas:
                omega_P = omega.centered_at_place(P)
                val1 = omega(a,b)
                val2 = omega_P(CC(0))
                self.assertLess(abs(val1-val2), 1e-8)

        # the places above x=oo are regular: P = (1/t, \pm 1/t**2 + O(1))
        # (however, all places at infinity are treated as discriminant)
        #
        # in this particular example, omega[0] = 1/(2*y). At the places at
        # infinity, these are equal to \mp 0.5, respectively. (the switch in
        # sign comes from the derivative dxdt = -1/t**2)
        places = X('oo')
        for P in places:
            sign = P.puiseux_series.ypart[-2]
            for omega in omegas:
                omega_P = omega.centered_at_place(P)
                val1 = -sign*0.5
                val2 = omega_P(CC(0))
                self.assertLess(abs(val1-val2), 1e-8)
