from .test_abelfunctions import AbelfunctionsTestCase
from abelfunctions.riemann_surface import RiemannSurface
from abelfunctions.abelmap import Jacobian, fractional_part

import numpy
import sympy

from numpy.linalg import norm
from sympy.abc import x,y

import unittest
import itertools


class TestJacobian(AbelfunctionsTestCase):

    def test_fractional_part(self):
        z = numpy.array([0,1,2,3])
        w1 = fractional_part(z)
        w2 = numpy.array([0,0,0,0])
        self.assertLess(norm(w1-w2), 1e-7)

        z = numpy.array([0.5,1.5,2.5,3.5])
        w1 = fractional_part(z)
        w2 = numpy.array([0.5,0.5,0.5,0.5])
        self.assertLess(norm(w1-w2), 1e-7)

        eps = 1e-12
        z = numpy.array([1-eps,1+eps])
        w1 = fractional_part(z)
        w2 = numpy.array([0,0])
        self.assertLess(norm(w1-w2), 1e-7)

        eps = 1e-8
        z = numpy.array([1-eps,1+eps])
        w1 = fractional_part(z)
        w2 = numpy.array([0,0])
        self.assertLess(norm(w1-w2), 1e-7)

    def test_already_reduced(self):
        g = self.X11.genus()
        J = Jacobian(self.X11)

        v = 2*numpy.random.rand(g)
        w = 3*numpy.random.rand(g)
        errorv = numpy.linalg.norm(J(v) - J(J(v)))
        errorw = numpy.linalg.norm(J(v) - J(J(v)))

        self.assertLess(errorv, 1e-14)
        self.assertLess(errorw, 1e-14)

    def test_lattice_vector(self):
        g = self.X11.genus()
        J = Jacobian(self.X11)
        Omega = self.X11.riemann_matrix()

        # create a random lattice vector
        alpha = numpy.random.randint(-5,5,g)
        beta = numpy.random.randint(-5,5,g)
        z = alpha + numpy.dot(Omega,beta)
        error = numpy.linalg.norm(J(z))
        self.assertLess(error, 1e-14)

    def test_half_lattice_vectors(self):
        g = self.X11.genus()
        J = Jacobian(self.X11)
        Omega = self.X11.riemann_matrix()

        # iterate over all possible half lattice vectors
        h1 = list(itertools.product((0,0.5),repeat=g))
        h2 = list(itertools.product((0,0.5),repeat=g))
        for hj in h1:
            hj = numpy.array(hj, dtype=numpy.complex)
            for hk in h2:
                hk = numpy.array(hk, dtype=numpy.complex)
                z = hj + numpy.dot(Omega,hk)
                error = numpy.linalg.norm(J(2*z))
                self.assertLess(error, 1e-14)
