from .test_abelfunctions import AbelfunctionsTestCase

import numpy

from abelfunctions import RiemannSurface
from abelfunctions.abelmap import AbelMap, Jacobian
from abelfunctions.differentials import Differential
from sage.rings.rational_field import QQ
from numpy.linalg import norm
from sage.rings.imaginary_unit import I


class TestDivisors(AbelfunctionsTestCase):
    def setUp(self):
        # cache some items for performance
        self.X11_Jacobian = Jacobian(self.X11)
        self.X11_P = self.X11(0)[0]
        self.X11_Q = self.X11(1)[0]
        self.X11_R = self.X11(I)[0]
        self.X11_P0 = self.X11.base_place

    def test_divisors_X11(self):
        J = self.X11_Jacobian
        P = self.X11_P
        D = 3 * P
        val1 = AbelMap(D)
        val2 = sum(ni * AbelMap(Pi) for (Pi, ni) in D)
        error = norm(J(val1 - val2))
        self.assertLess(error, 1e-7)

        D1 = sum(self.X11(2))
        val1 = AbelMap(D1)
        val2 = sum(ni * AbelMap(Pi) for (Pi, ni) in D1)
        error = norm(J(val1 - val2))
        self.assertLess(error, 1e-7)

        D2 = sum(self.X11(0.5))
        val1 = AbelMap(D2)
        val2 = sum(ni * AbelMap(Pi) for (Pi, ni) in D2)
        error = norm(J(val1 - val2))
        self.assertLess(error, 1e-7)

    def test_new_base_point_X11(self):
        J = self.X11_Jacobian
        P = self.X11_P
        Q = self.X11_Q
        R = self.X11_R
        self.X11_P0

        AP = AbelMap(P)
        AQ = AbelMap(Q)
        AR = AbelMap(R)

        APQ = AbelMap(P, Q)
        AQR = AbelMap(Q, R)
        ARP = AbelMap(R, P)

        error = norm(J(APQ - AQ + AP))
        self.assertLess(error, 1e-7)

        error = norm(J(AQR - AR + AQ))
        self.assertLess(error, 1e-7)

        error = norm(J(ARP - AP + AR))
        self.assertLess(error, 1e-7)


class TestJacobian(AbelfunctionsTestCase):
    def test_zero_X11(self):
        J = Jacobian(self.X11)
        zero = numpy.array([0, 0, 0, 0])
        value = J(zero)
        error = norm(value - zero)
        self.assertLess(error, 1e-14)


def test_close_discriminant():
    """Evaluate the AbelMap close to a discriminant point, see issue #115"""
    R = QQ["x,y"]
    x, y = R.gens()
    f = x**2 * y**3 - x**4 + 1
    X = RiemannSurface(f)
    omega = X.holomorphic_differentials()
    numer = omega[0].as_expression().numerator() + omega[2].as_expression().numerator()
    denom = omega[0].as_expression().denominator()
    eta = Differential(X, numer, denom)
    C = eta.valuation_divisor()
    AbelMap(C)
