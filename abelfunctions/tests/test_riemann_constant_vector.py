from .test_abelfunctions import AbelfunctionsTestCase

import abelfunctions
import numpy
import sympy
import unittest

from abelfunctions import (AbelMap, Jacobian, RiemannTheta,
                           RiemannConstantVector)
from abelfunctions.divisor import ZeroDivisor
from numpy.linalg import norm
from sympy import prod
from sympy.abc import x,y,z,t


class TestRCVTheta(AbelfunctionsTestCase):
    r"""Tests if the following theorem is satisfied:

    for any :math:`P \in X` and any effective degree :math:`g-1` divisor
    :math:`D`

    .. math::

        \theta(W,\Omega) \equiv 0

    if and only if

    .. math::

        W = A(P,D) + K(P)

    where :math:`A,K` are the Abel map and Riemann constant vector functions,
    respectively.
    """
    # helper function for Riemann theta function theorem test
    def is_theta_zero(self, X, W, prec=1e-7):
        Omega = X.riemann_matrix()
        J = Jacobian(X)
        W = J(W)
        u = RiemannTheta.oscillatory_part(W,Omega)
        self.assertLess(abs(u),prec)

    def test_theta_X11(self):
        P0 = self.X11.base_place
        W0 = RiemannConstantVector(P0)
        self.is_theta_zero(self.X11,W0)

        P_oo = self.X11('oo')[0]
        W_oo = RiemannConstantVector(P_oo)
        self.is_theta_zero(self.X11,W_oo)

        D = sum(self.X11(2))
        W_D = AbelMap(D) + RiemannConstantVector(P0)
        self.is_theta_zero(self.X11,W_D,prec=1e-6)

        # until AbelMap(P_oo,D) is implemented properly
        W_D_oo = AbelMap(P0,D) - D.degree*AbelMap(P_oo) + \
                 RiemannConstantVector(P_oo)
        self.is_theta_zero(self.X11,W_D_oo,prec=1e-6)

class TestRCVCanonical(AbelfunctionsTestCase):
    r"""Tests if the following theorem is satisfied:

    Let :math:`C` be a divisor of degree :math:`2g-2`. It is a canonical
    divisor if and only if

    .. math::

        2 K(P) \equiv - A(P,C)

    where :math:`A,K` are the Abel map and Riemann constant vector functions,
    respectively.
    """
    def test_degree_X11(self):
        g = self.X11.genus()
        assert g == 4

        oneforms = self.X11.holomorphic_differentials()
        degree = oneforms[0].valuation_divisor().degree
        self.assertEqual(degree,2*g-2)

        degree = oneforms[1].valuation_divisor().degree
        self.assertEqual(degree,2*g-2)

        degree = oneforms[2].valuation_divisor().degree
        self.assertEqual(degree,2*g-2)

        degree = oneforms[3].valuation_divisor().degree
        self.assertEqual(degree,2*g-2)

    def test_canonical_X11_0(self):
        X = self.X11
        J = Jacobian(X)
        g = X.genus()
        P0 = X.base_place
        oneforms = X.holomorphic_differentials()
        C = oneforms[0].valuation_divisor()
        W = 2*RiemannConstantVector(P0) + AbelMap(C)
        self.assertLess(norm(J(W)),1e-7)

    def test_canonical_X11_1(self):
        X = self.X11
        J = Jacobian(X)
        g = X.genus()
        P0 = X.base_place
        oneforms = X.holomorphic_differentials()
        C = oneforms[1].valuation_divisor()
        W = 2*RiemannConstantVector(P0) + AbelMap(C)
        self.assertLess(norm(J(W)),1e-7)

    def test_canonical_X11_2(self):
        X = self.X11
        J = Jacobian(X)
        g = X.genus()
        P0 = X.base_place
        oneforms = X.holomorphic_differentials()
        C = oneforms[2].valuation_divisor()
        W = 2*RiemannConstantVector(P0) + AbelMap(C)
        self.assertLess(norm(J(W)),1e-7)

    def test_canonical_X11_3(self):
        X = self.X11
        J = Jacobian(X)
        g = X.genus()
        P0 = X.base_place
        oneforms = X.holomorphic_differentials()
        C = oneforms[3].valuation_divisor()
        W = 2*RiemannConstantVector(P0) + AbelMap(C)
        self.assertLess(norm(J(W)),1e-7)

class TestErrors(AbelfunctionsTestCase):
    r"""Tests that certain errors are raised when incorrect input is given."""

    def test_degree_requirement(self):
        X = self.X11
        P = X.base_place
        C = ZeroDivisor(X)
        with self.assertRaises(ValueError):
            RiemannConstantVector(P,C=C)

    def test_same_surface_requirement(self):
        X11 = self.X11
        X2 = self.X2
        P = X11.base_place
        C = (2*X11.genus()-2)*X2.base_place # satisfies degree requirement
        with self.assertRaises(ValueError):
            RiemannConstantVector(P,C=C)
