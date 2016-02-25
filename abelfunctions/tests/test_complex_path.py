import unittest

from numpy import pi, Infinity, exp, sqrt
from abelfunctions.complex_path import (
    ComplexPathPrimitive,
    ComplexPath,
    ComplexLine,
    ComplexArc,
    ComplexRay,
)

class TestConstruction(unittest.TestCase):
    def test_line(self):
        gamma = ComplexLine(0, 1)
        self.assertEqual(gamma.x0, 0)
        self.assertEqual(gamma.x1, 1)

    def test_arc(self):
        gamma = ComplexArc(1, 0, pi, -pi)
        self.assertEqual(gamma.R, 1)
        self.assertEqual(gamma.w, 0)
        self.assertEqual(gamma.theta, pi)
        self.assertEqual(gamma.dtheta, -pi)

    def test_ray(self):
        gamma = ComplexRay(1)
        self.assertEqual(gamma.x0, 1)

    def test_composite(self):
        gamma1 = ComplexLine(-1, 0)
        gamma2 = ComplexLine(0, 1.j)
        gamma = gamma1 + gamma2
        self.assertEqual(gamma.segments, [gamma1, gamma2])

    def test_indexing(self):
        gamma0 = ComplexLine(-1, 0)
        gamma1 = ComplexLine(0, 1.j)
        gamma2 = ComplexArc(1, 0, pi/2, -pi/2)
        gamma = gamma0 + gamma1 + gamma2
        self.assertEqual(gamma.segments, [gamma0, gamma1, gamma2])
        self.assertEqual(gamma[0], gamma0)
        self.assertEqual(gamma[1], gamma1)
        self.assertEqual(gamma[2], gamma2)

    def test_iteration(self):
        gamma0 = ComplexLine(-1, 0)
        gamma1 = ComplexLine(0, 1.j)
        gamma2 = ComplexArc(1, 0, pi/2, -pi/2)
        gamma = gamma0 + gamma1 + gamma2

        index = 0
        for segment in gamma:
            if index == 0:
                self.assertEqual(segment, gamma0)
            elif index == 1:
                self.assertEqual(segment, gamma1)
            elif index == 2:
                self.assertEqual(segment, gamma2)
            index += 1

    def test_iteration_reverse(self):
        gamma0 = ComplexLine(-1, 0)
        gamma1 = ComplexLine(0, 1.j)
        gamma2 = ComplexArc(1, 0, pi/2, -pi/2)
        gamma = gamma0 + gamma1 + gamma2

        index = 0
        for segment in gamma[::-1]:
            if index == 0:
                self.assertEqual(segment, gamma2)
            elif index == 1:
                self.assertEqual(segment, gamma1)
            elif index == 2:
                self.assertEqual(segment, gamma0)
            index += 1

    def test_equality(self):
        gamma0 = ComplexLine(-1, 0)
        gamma1 = ComplexLine(-1, 0)
        self.assertEqual(gamma0, gamma1)

        gamma0 = ComplexArc(1, 0, 0, pi)
        gamma1 = ComplexArc(1, 0, 0, pi)
        self.assertEqual(gamma0, gamma1)

        gamma0 = ComplexRay(-1)
        gamma1 = ComplexRay(-1)
        self.assertEqual(gamma0, gamma1)

class TestReverse(unittest.TestCase):
    def test_single_line(self):
        gamma = ComplexLine(-1,2)
        gamma_rev = gamma.reverse()
        self.assertAlmostEqual(gamma(0.0), gamma_rev(1.0))
        self.assertAlmostEqual(gamma(0.1), gamma_rev(0.9))
        self.assertAlmostEqual(gamma(0.25), gamma_rev(0.75))
        self.assertAlmostEqual(gamma(0.50), gamma_rev(0.50))
        self.assertAlmostEqual(gamma(0.75), gamma_rev(0.25))
        self.assertAlmostEqual(gamma(1.0), gamma_rev(0.0))

    def test_single_arc(self):
        gamma = ComplexArc(1,0,pi/5,3*pi/5)
        gamma_rev = gamma.reverse()
        self.assertAlmostEqual(gamma(0.0), gamma_rev(1.0))
        self.assertAlmostEqual(gamma(0.1), gamma_rev(0.9))
        self.assertAlmostEqual(gamma(0.25), gamma_rev(0.75))
        self.assertAlmostEqual(gamma(0.50), gamma_rev(0.50))
        self.assertAlmostEqual(gamma(0.75), gamma_rev(0.25))
        self.assertAlmostEqual(gamma(1.0), gamma_rev(0.0))

    def test_composite(self):
        x1 = (sqrt(2)+sqrt(2)*1.j)/2
        gamma = ComplexLine(0,x1) + ComplexArc(1,0,pi/4,3*pi/5)
        gamma_rev = gamma.reverse()
        self.assertAlmostEqual(gamma(0.0), gamma_rev(1.0))
        self.assertAlmostEqual(gamma(0.1), gamma_rev(0.9))
        self.assertAlmostEqual(gamma(0.25), gamma_rev(0.75))
        self.assertAlmostEqual(gamma(0.50), gamma_rev(0.50))
        self.assertAlmostEqual(gamma(0.75), gamma_rev(0.25))
        self.assertAlmostEqual(gamma(1.0), gamma_rev(0.0))

class TestMismatchError(unittest.TestCase):
    def test_lines(self):
        gamma1 = ComplexLine(-1, 0)
        gamma2 = ComplexLine(42, 100)
        with self.assertRaises(ValueError):
            gamma1 + gamma2


class TestEvaluation(unittest.TestCase):
    def test_line(self):
        # using AlmostEqual for floating point error
        gamma = ComplexLine(0, 1)
        self.assertAlmostEqual(gamma(0), 0)
        self.assertAlmostEqual(gamma(0.5), 0.5)
        self.assertAlmostEqual(gamma(0.75), 0.75)
        self.assertAlmostEqual(gamma(1), 1)

        gamma = ComplexLine(-1.j, 1.j)
        self.assertAlmostEqual(gamma(0), -1.j)
        self.assertAlmostEqual(gamma(0.5), 0)
        self.assertAlmostEqual(gamma(0.75), 0.5j)
        self.assertAlmostEqual(gamma(1), 1.j)

    def test_arc(self):
        # arc from theta=0 to theta=pi/2 on the unit circle
        gamma = ComplexArc(1, 0, 0, pi/2)
        self.assertAlmostEqual(gamma(0), 1)
        self.assertAlmostEqual(gamma(0.5), exp(1.j*pi/4))
        self.assertAlmostEqual(gamma(0.75), exp(1.j*3*pi/8))
        self.assertAlmostEqual(gamma(1), exp(1.j*pi/2))

    def test_ray(self):
        # ray from x=-1 to infinity to the left
        gamma = ComplexRay(-1)
        self.assertAlmostEqual(gamma(0), -1)
        self.assertAlmostEqual(gamma(0.5), -2)
        self.assertAlmostEqual(gamma(0.75), -4)
        self.assertEqual(gamma(1), Infinity)

    def test_composite(self):
        gamma1 = ComplexLine(-1, 0)
        gamma2 = ComplexLine(0, 1.j)
        gamma = gamma1 + gamma2

        self.assertAlmostEqual(gamma(0), -1)
        self.assertAlmostEqual(gamma(0.25), -0.5)
        self.assertAlmostEqual(gamma(0.5), 0)
        self.assertAlmostEqual(gamma(0.75), 0.5j)
        self.assertAlmostEqual(gamma(1), 1.j)

class TestEvaluationDerivative(unittest.TestCase):
    def test_line_derivative(self):
        # using AlmostEqual for floating point error
        gamma = ComplexLine(0, 1)
        self.assertAlmostEqual(gamma.derivative(0), 1)
        self.assertAlmostEqual(gamma.derivative(0.5), 1)
        self.assertAlmostEqual(gamma.derivative(0.75), 1)
        self.assertAlmostEqual(gamma.derivative(0), 1)

        gamma = ComplexLine(-1.j, 1.j)
        self.assertAlmostEqual(gamma.derivative(0), 2.j)
        self.assertAlmostEqual(gamma.derivative(0.5), 2.j)
        self.assertAlmostEqual(gamma.derivative(0.75), 2.j)
        self.assertAlmostEqual(gamma.derivative(1), 2.j)

    def test_arc_derivative(self):
        # arc from theta=0 to theta=pi/2 on the unit circle
        gamma = ComplexArc(1, 0, 0, pi/2)
        scale = 1.j*pi/2
        self.assertAlmostEqual(gamma.derivative(0), scale)
        self.assertAlmostEqual(gamma.derivative(0.5), scale*exp(1.j*pi/4))
        self.assertAlmostEqual(gamma.derivative(0.75), scale*exp(1.j*3*pi/8))
        self.assertAlmostEqual(gamma.derivative(1), scale*exp(1.j*pi/2))

    def test_ray_derivative(self):
        # ray from x=-1 to infinity to the left
        gamma = ComplexRay(-1)
        self.assertAlmostEqual(gamma.derivative(0), 1)
        self.assertAlmostEqual(gamma.derivative(0.5), 4)
        self.assertAlmostEqual(gamma.derivative(0.75), 16)
        self.assertAlmostEqual(gamma.derivative(1), Infinity)

    def test_composite(self):
        gamma1 = ComplexLine(-1, 0)  # derivative == 1
        gamma2 = ComplexLine(0, 1.j) # derivative == 1.j
        gamma = gamma1 + gamma2

        # derivative is defined on the half-open intervals [s_i,s_{i+1}) except
        # for the last segment
        self.assertAlmostEqual(gamma.derivative(0), 1)
        self.assertAlmostEqual(gamma.derivative(0.25), 1)
        self.assertAlmostEqual(gamma.derivative(0.49), 1)
        self.assertAlmostEqual(gamma.derivative(0.5), 1.j)
        self.assertAlmostEqual(gamma.derivative(0.51), 1.j)
        self.assertAlmostEqual(gamma.derivative(0.75), 1.j)
        self.assertAlmostEqual(gamma.derivative(1), 1.j)
