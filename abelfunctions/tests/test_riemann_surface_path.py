import unittest

import numpy
from numpy import pi, Infinity, exp, sqrt, power, complex
from sage.all import QQ, QQbar, e, I

from abelfunctions.complex_path import (
    ComplexLine,
    ComplexArc,
    ComplexRay,
)
from abelfunctions.puiseux import puiseux
from abelfunctions.riemann_surface import RiemannSurface
from abelfunctions.riemann_surface_path import(
    ordered_puiseux_series,
    RiemannSurfacePathPuiseux,
    RiemannSurfacePathSmale,
)


class TestOrderedPuiseuxSeries(unittest.TestCase):
    def setUp(self):
        R = QQ['x,y']
        x,y = R.gens()

        f1 = y**2 - x
        self.f1 = f1
        self.X1 = RiemannSurface(f1)

        f2 = y**3 - x
        self.f2 = f2
        self.X2 = RiemannSurface(f2)

    def test_example_puiseux(self):
        p = puiseux(self.f1, 0)[0]
        px = p.xseries()
        R = px[0].parent()
        x = R.gen()
        half = QQ(1)/2
        self.assertEqual(px[0].truncate(1), -x**half)
        self.assertEqual(px[1].truncate(1), x**half)

        p = puiseux(self.f2, 0)[0]
        px = p.xseries()
        R = px[0].parent()
        x = R.gen()
        third = QQ(1)/3
        S = QQ['t']; t = S.gen()
        alpha,beta,gamma = (t**3 - 1).roots(ring=QQbar, multiplicities=False)
        self.assertEqual(px[0].truncate(1), alpha*x**third)
        self.assertEqual(px[1].truncate(1), gamma*x**third)
        self.assertEqual(px[2].truncate(1), beta*x**third)

    def test_example_puiseux_extend(self):
        p = puiseux(self.f1, 0)[0]
        p.extend(10)
        px = p.xseries()
        R = px[0].parent()
        x = R.gen()
        half = QQ(1)/2
        self.assertEqual(px[0].truncate(9), -x**half)
        self.assertEqual(px[1].truncate(9), x**half)

        p = puiseux(self.f2, 0)[0]
        p.extend(10)
        px = p.xseries()
        R = px[0].parent()
        x = R.gen()
        third = QQ(1)/3
        S = QQ['t']; t = S.gen()
        alpha,beta,gamma = (t**3 - 1).roots(ring=QQbar, multiplicities=False)
        self.assertEqual(px[0].truncate(9), alpha*x**third)
        self.assertEqual(px[1].truncate(9), gamma*x**third)
        self.assertEqual(px[2].truncate(9), beta*x**third)

    def test_example_puiseux_extend_to_x(self):
        p = puiseux(self.f1, 0)[0]
        p.extend_to_x(0.5)
        px = p.xseries()
        R = px[0].parent()
        x = R.gen()
        half = QQ(1)/2
        self.assertEqual(px[0].truncate(2), -x**half)
        self.assertEqual(px[1].truncate(2), x**half)

        p = puiseux(self.f2, 0)[0]
        p.extend_to_x(0.5)
        px = p.xseries()
        R = px[0].parent()
        x = R.gen()
        third = QQ(1)/3
        S = QQ['t']; t = S.gen()
        alpha,beta,gamma = (t**3 - 1).roots(ring=QQbar, multiplicities=False)
        self.assertEqual(px[0].truncate(2), alpha*x**third)
        self.assertEqual(px[1].truncate(2), gamma*x**third)
        self.assertEqual(px[2].truncate(2), beta*x**third)

    def test_ordered_puiseux_series_discriminant(self):
        # testing f1
        target_point = 0
        gammax = ComplexLine(1, target_point)
        y0 = [-1,1]
        p, place = ordered_puiseux_series(self.X1, gammax, y0, target_point)
        P = p[0].parent()
        x = P.gen()
        half = QQ(1)/2
        self.assertEqual(p[0].truncate(1), -x**half)
        self.assertEqual(p[1].truncate(1), x**half)

        y0 = [1,-1]
        p, place = ordered_puiseux_series(self.X1, gammax, y0, target_point)
        self.assertEqual(p[0].truncate(1), x**half)
        self.assertEqual(p[1].truncate(1), -x**half)

        # testing f2
        S = QQ['t']; t = S.gen()
        alpha,beta,gamma = (t**3 - 1).roots(ring=QQbar, multiplicities=False)
        third = QQ(1)/3
        y0 = [alpha, beta, gamma]
        p, place = ordered_puiseux_series(self.X2, gammax, y0, target_point)
        self.assertEqual(p[0].truncate(1), alpha*x**third)
        self.assertEqual(p[1].truncate(1), beta*x**third)
        self.assertEqual(p[2].truncate(1), gamma*x**third)

        y0 = [beta, gamma, alpha]
        p, place = ordered_puiseux_series(self.X2, gammax, y0, target_point)
        self.assertEqual(p[0].truncate(1), beta*x**third)
        self.assertEqual(p[1].truncate(1), gamma*x**third)
        self.assertEqual(p[2].truncate(1), alpha*x**third)

        y0 = [beta, alpha, gamma]
        p, place = ordered_puiseux_series(self.X2, gammax, y0, target_point)
        self.assertEqual(p[0].truncate(1), beta*x**third)
        self.assertEqual(p[1].truncate(1), alpha*x**third)
        self.assertEqual(p[2].truncate(1), gamma*x**third)

    def test_ordered_puiseux_series_regular(self):
        # testing f1
        target_point = 4
        gammax = ComplexLine(1, target_point)
        y0 = [-1,1]
        p, place = ordered_puiseux_series(self.X1, gammax, y0, target_point)
        P = p[0].parent()
        x = P.gen()
        self.assertEqual(p[0].truncate(3), -2 - QQ(1)/4*x + QQ(1)/64*x**2)
        self.assertEqual(p[1].truncate(3), 2 + QQ(1)/4*x - QQ(1)/64*x**2)

        y0 = [1,-1]
        p, place = ordered_puiseux_series(self.X1, gammax, y0, target_point)
        self.assertEqual(p[0].truncate(3), 2 + QQ(1)/4*x - QQ(1)/64*x**2)
        self.assertEqual(p[1].truncate(3), -2 - QQ(1)/4*x + QQ(1)/64*x**2)


class TestRiemannSurfacePathPuiseux(unittest.TestCase):
    def setUp(self):
        R = QQ['x,y']
        x,y = R.gens()

        f1 = y**2 - x
        self.f1 = f1
        self.X1 = RiemannSurface(f1)

        f2 = y**3 - x
        self.f2 = f2
        self.X2 = RiemannSurface(f2)

    def test_construction(self):
        gammax = ComplexLine(1,0)
        y0 = [-1,1]
        gamma = RiemannSurfacePathPuiseux(self.X1, gammax, y0)

    def test_analytic_continuation_X1(self):
        gammax = ComplexLine(1,0)
        y0 = [-1,1]
        gamma = RiemannSurfacePathPuiseux(self.X1, gammax, y0)

        y = gamma.get_y(0)
        self.assertAlmostEqual(y[0], -1)
        self.assertAlmostEqual(y[1], 1)

        y = gamma.get_y(0.5)
        self.assertAlmostEqual(y[0], -sqrt(complex(0.5)))
        self.assertAlmostEqual(y[1], sqrt(complex(0.5)))

        y = gamma.get_y(0.75)
        self.assertAlmostEqual(y[0], -sqrt(complex(0.25)))
        self.assertAlmostEqual(y[1], sqrt(complex(0.25)))

        y = gamma.get_y(1)
        self.assertAlmostEqual(y[0], 0)
        self.assertAlmostEqual(y[1], 0)

        gammax = ComplexArc(2,2,0,pi)
        y0 = [-2,2]
        gamma = RiemannSurfacePathPuiseux(self.X1, gammax, y0)

        y = gamma.get_y(0)
        self.assertAlmostEqual(y[0], -2)
        self.assertAlmostEqual(y[1], 2)

        y = gamma.get_y(1)
        self.assertAlmostEqual(y[0], 0)
        self.assertAlmostEqual(y[1], 0)

    def test_analytic_continuation_X2(self):
        S = QQ['t']; t = S.gen()
        a,b,c = (t**3 - 1).roots(ring=QQbar, multiplicities=False)

        gammax = ComplexLine(1,0)
        y0 = [a,b,c]
        gamma = RiemannSurfacePathPuiseux(self.X2, gammax, y0)

        y = gamma.get_y(0)
        self.assertAlmostEqual(y[0], a)
        self.assertAlmostEqual(y[1], b)
        self.assertAlmostEqual(y[2], c)

        scale = (0.5)**(1/3.)
        y = gamma.get_y(0.5)
        self.assertAlmostEqual(y[0], scale*a)
        self.assertAlmostEqual(y[1], scale*b)
        self.assertAlmostEqual(y[2], scale*c)

        y = gamma.get_y(1)
        self.assertAlmostEqual(y[0], 0)
        self.assertAlmostEqual(y[1], 0)
        self.assertAlmostEqual(y[2], 0)

    def test_rays(self):
        # test that analytic continuation to places at infinity work
        gammax = ComplexRay(-9)
        y0 = [-3.j,3.j]
        gamma = RiemannSurfacePathPuiseux(self.X1, gammax, y0)

        y = gamma.get_y(0)
        self.assertAlmostEqual(y[0], -3.j)
        self.assertAlmostEqual(y[1], 3.j)

        # note: the infinity behavior may change in the future
        y = gamma.get_y(1)
        self.assertTrue(numpy.isnan(y[0]))
        self.assertTrue(numpy.isnan(y[1]))


class TestRiemannSurfacePathSmale(unittest.TestCase):
    def setUp(self):
        R = QQ['x,y']
        x,y = R.gens()

        f1 = y**2 - x
        self.f1 = f1
        self.X1 = RiemannSurface(f1)

        f2 = y**3 - x
        self.f2 = f2
        self.X2 = RiemannSurface(f2)

    def test_construction(self):
        gammax = ComplexLine(1,4)
        y0 = [-1,1]
        gamma = RiemannSurfacePathSmale(self.X1, gammax, y0)

    def test_analytic_continuation_X1(self):
        gammax = ComplexLine(1,4)
        y0 = [-1,1]
        gamma = RiemannSurfacePathSmale(self.X1, gammax, y0)

        y = gamma.get_y(0)
        self.assertAlmostEqual(y[0], -1)
        self.assertAlmostEqual(y[1], 1)

        y = gamma.get_y(0.5)
        self.assertAlmostEqual(y[0], -sqrt(2.5))
        self.assertAlmostEqual(y[1], sqrt(2.5))

        y = gamma.get_y(0.75)
        self.assertAlmostEqual(y[0], -sqrt(3.25))
        self.assertAlmostEqual(y[1], sqrt(3.25))

        y = gamma.get_y(1)
        self.assertAlmostEqual(y[0], -2)
        self.assertAlmostEqual(y[1], 2)

    def test_analytic_continuation_X1_big_jump(self):
        # tests that smale will handle the case when checkpoints don't exist or
        # are far away from each other
        gammax = ComplexLine(1,9)
        y0 = [-1,1]
        gamma = RiemannSurfacePathSmale(self.X1, gammax, y0,
                                        ncheckpoints=1)

        y = gamma.get_y(0)
        self.assertAlmostEqual(y[0], -1)
        self.assertAlmostEqual(y[1], 1)

        y = gamma.get_y(0.5)
        self.assertAlmostEqual(y[0], -sqrt(5))
        self.assertAlmostEqual(y[1], sqrt(5))

        y = gamma.get_y(0.75)
        self.assertAlmostEqual(y[0], -sqrt(7))
        self.assertAlmostEqual(y[1], sqrt(7))

        y = gamma.get_y(1)
        self.assertAlmostEqual(y[0], -3)
        self.assertAlmostEqual(y[1], 3)

    def tests_monodromy(self):
        gammax = ComplexArc(1, 0, 0, 2*pi)
        y0 = [-1,1]
        gamma = RiemannSurfacePathSmale(self.X1, gammax, y0)

        y = gamma.get_y(0.0)
        self.assertAlmostEqual(y[0], -1)
        self.assertAlmostEqual(y[1], 1)

        y = gamma.get_y(1.0)
        self.assertAlmostEqual(y[0], 1)
        self.assertAlmostEqual(y[1], -1)


class TestRiemannSurfacePathComposite(unittest.TestCase):
    def setUp(self):
        R = QQ['x,y']
        x,y = R.gens()

        f1 = y**2 - x
        self.f1 = f1
        self.X1 = RiemannSurface(f1)

        f2 = y**3 - x
        self.f2 = f2
        self.X2 = RiemannSurface(f2)

    def test_singleton_segment(self):
        gamma1x = ComplexLine(1,4)
        y01 = [-1,1]
        gamma1 = RiemannSurfacePathSmale(self.X1, gamma1x, y01)
        self.assertEqual(gamma1.segments[0], gamma1)

        gamma2x = ComplexLine(4,9)
        y02 = [-2,2]
        gamma2 = RiemannSurfacePathSmale(self.X1, gamma2x, y02)
        self.assertEqual(gamma2.segments[0], gamma2)

    def test_segments(self):
        gamma1x = ComplexLine(1,4)
        y01 = [-1,1]
        gamma1 = RiemannSurfacePathSmale(self.X1, gamma1x, y01)

        gamma2x = ComplexLine(4,9)
        y02 = [-2,2]
        gamma2 = RiemannSurfacePathSmale(self.X1, gamma2x, y02)

        gamma = gamma1 + gamma2
        self.assertEqual(gamma.segments[0], gamma1)
        self.assertEqual(gamma.segments[1], gamma2)
        self.assertEqual(gamma[0], gamma1)
        self.assertEqual(gamma[1], gamma2)

    def test_get_x(self):
        gamma1x = ComplexLine(1,4)
        y01 = [-1,1]
        gamma1 = RiemannSurfacePathSmale(self.X1, gamma1x, y01)

        gamma2x = ComplexLine(4,9)
        y02 = [-2,2]
        gamma2 = RiemannSurfacePathSmale(self.X1, gamma2x, y02)

        gamma = gamma1 + gamma2

        x = gamma.get_x(0)
        self.assertAlmostEqual(x, 1)

        x = gamma.get_x(0.25)
        self.assertAlmostEqual(x, 2.5)

        x = gamma.get_x(0.5)
        self.assertAlmostEqual(x, 4)

        x = gamma.get_x(0.75)
        self.assertAlmostEqual(x, 6.5)

        x = gamma.get_x(1.0)
        self.assertAlmostEqual(x, 9)

    def test_analytic_continuation(self):
        # method 1: adding two RSPs
        gamma1x = ComplexLine(1,4)
        y01 = [-1,1]
        gamma1 = RiemannSurfacePathSmale(self.X1, gamma1x, y01)

        gamma2x = ComplexLine(4,9)
        y02 = [-2,2]
        gamma2 = RiemannSurfacePathSmale(self.X1, gamma2x, y02)

        gamma = gamma1 + gamma2

        y = gamma.get_y(0)
        self.assertAlmostEqual(y[0], -1)
        self.assertAlmostEqual(y[1], 1)

        y = gamma.get_y(0.25)
        self.assertAlmostEqual(y[0], -sqrt(2.5))
        self.assertAlmostEqual(y[1], sqrt(2.5))

        y = gamma.get_y(0.5)
        self.assertAlmostEqual(y[0], -2)
        self.assertAlmostEqual(y[1], 2)

        y = gamma.get_y(0.75)
        self.assertAlmostEqual(y[0], -sqrt(6.5))
        self.assertAlmostEqual(y[1], sqrt(6.5))

        y = gamma.get_y(1.0)
        self.assertAlmostEqual(y[0], -3)
        self.assertAlmostEqual(y[1], 3)

        # method 2: composite ComplexPath with one RSP
        gammax = gamma1x + gamma2x
        y0 = [-1,1]
        gamma = RiemannSurfacePathSmale(self.X1, gammax, y0)

        y = gamma.get_y(0)
        self.assertAlmostEqual(y[0], -1)
        self.assertAlmostEqual(y[1], 1)

        y = gamma.get_y(0.25)
        self.assertAlmostEqual(y[0], -sqrt(2.5))
        self.assertAlmostEqual(y[1], sqrt(2.5))

        y = gamma.get_y(0.5)
        self.assertAlmostEqual(y[0], -2)
        self.assertAlmostEqual(y[1], 2)

        y = gamma.get_y(0.75)
        self.assertAlmostEqual(y[0], -sqrt(6.5))
        self.assertAlmostEqual(y[1], sqrt(6.5))

        y = gamma.get_y(1.0)
        self.assertAlmostEqual(y[0], -3)
        self.assertAlmostEqual(y[1], 3)

    def test_addition_fails(self):
        # case 1: the x-points don't match
        gamma1x = ComplexLine(1,4)
        y01 = [-1,1]
        gamma1 = RiemannSurfacePathSmale(self.X1, gamma1x, y01)

        gamma2x = ComplexLine(9,10)
        y02 = [-3,3]
        gamma2 = RiemannSurfacePathSmale(self.X1, gamma2x, y02)

        with self.assertRaises(ValueError):
            gamma = gamma1 + gamma2

        # case 2: x-points match but y-fibre doesn't
        gamma1x = ComplexLine(1,4)
        y01 = [-1,1]
        gamma1 = RiemannSurfacePathSmale(self.X1, gamma1x, y01)

        gamma2x = ComplexLine(4,9)
        y02 = [2,-2]  # swapped: gamma1 ends at [-2,2]
        gamma2 = RiemannSurfacePathSmale(self.X1, gamma2x, y02)

        with self.assertRaises(ValueError):
            gamma = gamma1 + gamma2


class TestParameterize(unittest.TestCase):
    def setUp(self):
        R = QQ['x,y']
        x,y = R.gens()

        f1 = y**2 - x
        self.f1 = f1
        self.X1 = RiemannSurface(f1)

        f2 = y**3 - x
        self.f2 = f2
        self.X2 = RiemannSurface(f2)

    def test_simple_line_smale(self):
        gammax = ComplexLine(1,2)
        gamma = RiemannSurfacePathSmale(self.X1, gammax, [-1,1])
        nu = lambda x,y: y
        nu_gamma = gamma.parameterize(nu)

        val = nu_gamma(0.0)
        self.assertAlmostEqual(val, -1)

        val = nu_gamma(0.5)
        self.assertAlmostEqual(val, -sqrt(1.5))

        val = nu_gamma(1.0)
        self.assertAlmostEqual(val, -sqrt(2.0))

    @unittest.skip('Skip until differentials and localization are worked out.')
    def test_simple_line_puiseux_discriminant(self):
        gammax = ComplexLine(2,0) #dxds = 2
        y0 =  [-sqrt(2.0),sqrt(2.0)]
        gamma = RiemannSurfacePathPuiseux(self.X1, gammax, y0)
        nu = lambda x,y: y
        nu_gamma = gamma.parameterize(nu)

        val = nu_gamma(0.0)
        self.assertAlmostEqual(val, -sqrt(2.0))

        val = nu_gamma(0.5)
        self.assertAlmostEqual(val, -sqrt(1.0))

        val = nu_gamma(1.0)
        self.assertAlmostEqual(val, 0)

    def test_simple_line_dxds(self):
        gammax = ComplexLine(1,3)  # dx/ds = 2
        gamma = RiemannSurfacePathSmale(self.X1, gammax, [-1,1])
        nu = lambda x,y: y
        nu_gamma = gamma.parameterize(nu)

        val = nu_gamma(0.0)
        self.assertAlmostEqual(val, -2)

        val = nu_gamma(0.5)
        self.assertAlmostEqual(val, -2*sqrt(2.0))

        val = nu_gamma(1.0)
        self.assertAlmostEqual(val, -2*sqrt(3.0))

    def test_simple_arc(self):
        gammax = ComplexArc(1,0,0,pi)
        gamma = RiemannSurfacePathSmale(self.X1, gammax, [-1,1])
        nu = lambda x,y: y
        nu_gamma = gamma.parameterize(nu)

        val = nu_gamma(0.0)
        test = gammax.derivative(0.0)*(-1)
        self.assertAlmostEqual(val, test)

        val = nu_gamma(0.5)
        test = gammax.derivative(0.5)*(-sqrt(1.j))
        self.assertAlmostEqual(val, test)

        val = nu_gamma(1.0)
        test = gammax.derivative(1.0)*(-1.j)
        self.assertAlmostEqual(val, test)

    def test_simple_composite(self):
        gammax1 = ComplexLine(4,1)
        gamma1 = RiemannSurfacePathSmale(self.X1, gammax1, [-2,2])
        gammax2 = ComplexArc(1,0,0,pi)
        gamma2 = RiemannSurfacePathSmale(self.X1, gammax2, [-1,1])
        gamma = gamma1 + gamma2
        nu = lambda x,y: y
        nu_gamma = gamma.parameterize(nu)

        val = nu_gamma(0.0)
        test = gammax1.derivative(0.0)*(-2)
        self.assertAlmostEqual(val, test)

        val = nu_gamma(0.25)
        test = gammax1.derivative(0.5)*(-sqrt(2.5))
        self.assertAlmostEqual(val, test)

        eps = 1e-12
        val = nu_gamma(0.5-eps)
        test = gammax1.derivative(1.0-eps/2)*(-1)
        self.assertAlmostEqual(val, test)

        val = nu_gamma(0.5)
        test = gammax2.derivative(0.0)*(-1)
        self.assertAlmostEqual(val, test)

        val = nu_gamma(0.5+eps)
        test = gammax2.derivative(eps/2)*(-1)
        self.assertAlmostEqual(val, test)

        val = nu_gamma(0.75)
        test = gammax2.derivative(0.5)*(-sqrt(1.j))
        self.assertAlmostEqual(val, test)

        val = nu_gamma(1.0)
        test = gammax2.derivative(1.0)*(-1.j)
        self.assertAlmostEqual(val, test)
