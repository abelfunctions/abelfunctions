import unittest
import numpy

from abelfunctions.riemann_theta.radius import (
    radius,
    radius0,
    radius1,
    radius2,
    radiusN,
)

class TestRadiusN(unittest.TestCase):
    def test_vs_radius1_1e8(self):
        # genus 3 example
        z = numpy.array([0.2+0.5j, 0.3-0.1j, -0.1+0.2j], dtype=numpy.complex)
        T = numpy.diag([1,2,3]) + numpy.diag([4,5], k=1) + numpy.diag([6], k=2)
        rho = 0.9  # any rho is fine
        eps = 1e-8

        deriv = [1,0,0]
        R1 = radius1(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, [deriv])
        self.assertAlmostEqual(R1, R2)

        deriv = [0,1,0]
        R1 = radius1(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, [deriv])
        self.assertAlmostEqual(R1, R2)

        deriv = [0,0,1]
        R1 = radius1(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, [deriv])
        self.assertAlmostEqual(R1, R2)

        deriv = [0.1 + 0.2j, 0.3+0.4j, 0.5+0.6j]
        R1 = radius1(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, [deriv])
        self.assertAlmostEqual(R1, R2)

    def test_vs_radius1_1e14(self):
        # genus 3 example
        z = numpy.array([0.2+0.5j, 0.3-0.1j, -0.1+0.2j], dtype=numpy.complex)
        T = numpy.diag([1,2,3]) + numpy.diag([4,5], k=1) + numpy.diag([6], k=2)
        rho = 0.9  # any rho is fine
        eps = 1e-14

        deriv = [1,0,0]
        R1 = radius1(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, [deriv])
        self.assertAlmostEqual(R1, R2)

        deriv = [0,1,0]
        R1 = radius1(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, [deriv])
        self.assertAlmostEqual(R1, R2)

        deriv = [0,0,1]
        R1 = radius1(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, [deriv])
        self.assertAlmostEqual(R1, R2)

        deriv = [0.1 + 0.2j, 0.3+0.4j, 0.5+0.6j]
        R1 = radius1(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, [deriv])
        self.assertAlmostEqual(R1, R2)

    def test_vs_radius2_1e8(self):
        # genus 3 example
        z = numpy.array([0.2+0.5j, 0.3-0.1j, -0.1+0.2j], dtype=numpy.complex)
        T = numpy.diag([1,2,3]) + numpy.diag([4,5], k=1) + numpy.diag([6], k=2)
        rho = 0.9  # any rho is fine
        eps = 1e-8

        deriv = [[1,0,0], [1,0,0]]
        R1 = radius2(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, deriv)
        self.assertAlmostEqual(R1, R2)

        deriv = [[0,1,0], [1,0,0]]
        R1 = radius2(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, deriv)
        self.assertAlmostEqual(R1, R2)

        deriv = [[0,0,1], [1,0,0]]
        R1 = radius2(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, deriv)
        self.assertAlmostEqual(R1, R2)

        deriv = [[1,0,0], [0,1,0]]
        R1 = radius2(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, deriv)
        self.assertAlmostEqual(R1, R2)

        deriv = [[0,1,0], [0,1,0]]
        R1 = radius2(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, deriv)
        self.assertAlmostEqual(R1, R2)

        deriv = [[0,0,1], [0,1,0]]
        R1 = radius2(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, deriv)
        self.assertAlmostEqual(R1, R2)

        deriv = [[1,0,0], [0,0,1]]
        R1 = radius2(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, deriv)
        self.assertAlmostEqual(R1, R2)

        deriv = [[0,1,0], [0,0,1]]
        R1 = radius2(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, deriv)
        self.assertAlmostEqual(R1, R2)

        deriv = [[0,0,1], [0,0,1]]
        R1 = radius2(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, deriv)
        self.assertAlmostEqual(R1, R2)

        deriv = [[0.1 + 0.2j, 0.3+0.4j, 0.5+0.6j],
                 [0.7 + 0.8j, 0.9+1.0j, 1.1+1.2j]]
        R1 = radius2(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, deriv)
        self.assertAlmostEqual(R1, R2)

    def test_vs_radius2_1e14(self):
        # genus 3 example
        z = numpy.array([0.2+0.5j, 0.3-0.1j, -0.1+0.2j], dtype=numpy.complex)
        T = numpy.diag([1,2,3]) + numpy.diag([4,5], k=1) + numpy.diag([6], k=2)
        rho = 0.9  # any rho is fine
        eps = 1e-14

        deriv = [[1,0,0], [1,0,0]]
        R1 = radius2(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, deriv)
        self.assertAlmostEqual(R1, R2)

        deriv = [[0,1,0], [1,0,0]]
        R1 = radius2(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, deriv)
        self.assertAlmostEqual(R1, R2)

        deriv = [[0,0,1], [1,0,0]]
        R1 = radius2(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, deriv)
        self.assertAlmostEqual(R1, R2)

        deriv = [[1,0,0], [0,1,0]]
        R1 = radius2(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, deriv)
        self.assertAlmostEqual(R1, R2)

        deriv = [[0,1,0], [0,1,0]]
        R1 = radius2(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, deriv)
        self.assertAlmostEqual(R1, R2)

        deriv = [[0,0,1], [0,1,0]]
        R1 = radius2(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, deriv)
        self.assertAlmostEqual(R1, R2)

        deriv = [[1,0,0], [0,0,1]]
        R1 = radius2(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, deriv)
        self.assertAlmostEqual(R1, R2)

        deriv = [[0,1,0], [0,0,1]]
        R1 = radius2(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, deriv)
        self.assertAlmostEqual(R1, R2)

        deriv = [[0,0,1], [0,0,1]]
        R1 = radius2(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, deriv)
        self.assertAlmostEqual(R1, R2)

        deriv = [[0.1 + 0.2j, 0.3+0.4j, 0.5+0.6j],
                 [0.7 + 0.8j, 0.9+1.0j, 1.1+1.2j]]
        R1 = radius2(eps, rho, 3, T, deriv)
        R2 = radiusN(eps, rho, 3, T, deriv)
        self.assertAlmostEqual(R1, R2)

