from .test_abelfunctions import AbelfunctionsTestCase
from abelfunctions.riemann_surface import RiemannSurface
from abelfunctions.abelmap import Jacobian

import numpy
import sympy
from sympy.abc import x,y

import unittest
import itertools


class TestJacobian(unittest.TestCase):

    def setUp(self):
        self.f = x**2*y**3 - x**4 + 1
        self.X = RiemannSurface(self.f,x,y)
        self.g = self.X.genus()
        self.J = Jacobian(self.X)

    def test_already_reduced(self):
        J = self.J

        v = 2*numpy.random.rand(self.g,1)
        w = 3*numpy.random.rand(self.g,1)
        errorv = numpy.linalg.norm(J(v) - J(J(v)))
        errorw = numpy.linalg.norm(J(v) - J(J(v)))

        self.assertLess(errorv, 1e-14)
        self.assertLess(errorw, 1e-14)

    def test_lattice_vector(self):
        J = self.J
        Omega = self.J.Omega

        # create a random lattice vector
        alpha = numpy.random.randint(-5,5,size=(self.g,1))
        beta = numpy.random.randint(-5,5,size=(self.g,1))
        z = alpha + numpy.dot(Omega,beta)
        error = numpy.linalg.norm(J(z))
        self.assertLess(error, 1e-14)

    def test_half_lattice_vectors(self):
        J = self.J
        Omega = self.J.Omega

        # iterate over all possible half lattice vectors
        h1 = list(itertools.product((0,0.5),repeat=self.g))
        h2 = list(itertools.product((0,0.5),repeat=self.g))
        for hj in h1:
            hj = numpy.array(hj, dtype=numpy.complex).reshape((self.g,1))
            for hk in h2:
                hk = numpy.array(hk, dtype=numpy.complex).reshape((self.g,1))
                z = hj + numpy.dot(Omega,hk)
                error = numpy.linalg.norm(J(2*z))
                self.assertLess(error, 1e-14)
