import unittest
import six

from abelfunctions.puiseux import (
    almost_monicize,
    newton_data,
    newton_iteration,
    newton_polygon,
    newton_polygon_exceptional,
    puiseux,
    puiseux_rational,
    transform_newton_polynomial,
)
from abelfunctions.puiseux_series_ring import PuiseuxSeriesRing
from abelfunctions.tests.test_abelfunctions import AbelfunctionsTestCase

from sage.all import SR, xgcd
from sage.calculus.functional import taylor
from sage.calculus.var import var
from sage.rings.laurent_series_ring import LaurentSeriesRing
from sage.rings.rational_field import QQ
from sage.rings.qqbar import QQbar
from sage.rings.infinity import infinity
from sympy import Poly, Point, Segment, Polygon, RootOf, sqrt, S

# every example will be over QQ[x,y]. consider putting in setup?
R = QQ['x,y']
S = QQ['t']
x,y = R.gens()
t = S.gens()

class TestNewtonPolygon(unittest.TestCase):

    def test_segment(self):
        H = y + x
        self.assertEqual(newton_polygon(H),
                         [[(0,1),(1,0)]])

        H = y**2 + x**2
        self.assertEqual(newton_polygon(H),
                         [[(0,2),(2,0)]])

    def test_general_segment(self):
        H = y**2 + x**4
        self.assertEqual(newton_polygon(H),
                         [[(0,2),(2,0)]])

    def test_colinear(self):
        H = x**4 + x**2*y**2 + y**4
        self.assertEqual(newton_polygon(H),
                         [[(0,4),(2,2),(4,0)]])

        H = x**4 + x**2*y**2 + x*y**3 + y**4
        self.assertEqual(newton_polygon(H),
                         [[(0,4),(2,2),(3,1),(4,0)]])

    def test_multiple(self):
        H = 2*x**2 + 3*x*y + 5*y**3
        self.assertEqual(newton_polygon(H),
                         [[(0,2),(1,1)], [(1,1),(3,0)]])

    def test_general_to_colinear(self):
        H = x**5 + x**2*y**2 + y**4
        self.assertEqual(newton_polygon(H),
                         [[(0,4),(2,2),(4,0)]])

        H = x**5 + x**3*y + x**2*y**2 + y**4
        self.assertEqual(newton_polygon(H),
                         [[(0,4),(1,3),(2,2),(4,0)]])

        H = x**5 + x**2*y**2 + x*y**3 + y**4
        self.assertEqual(newton_polygon(H),
                         [[(0,4),(2,2),(3,1),(4,0)]])

    def test_exceptional(self):
        H = x**5 + x**2*y**2 + y**4
        EN = newton_polygon_exceptional(H)
        self.assertEqual(EN, [[(0,0),(4,0)]])

        # issue 111
        H = -x**8 + y**4 + x*y**2
        EN = newton_polygon_exceptional(H)
        self.assertEqual(EN, [[(0,0),(4,0)]])

    def test_issue111(self):
        H = -x**8 + y**4 + x*y**2
        N = newton_polygon(H)

class TestNewtonData(unittest.TestCase):

    def test_segment(self):
        H = 2*x + 3*y
        self.assertEqual(newton_data(H),
                         [(1, 1, 1, 3*x + 2)])

        H = 2*x**2 + 3*y**2
        self.assertEqual(newton_data(H),
                         [(1, 1, 2, 3*x**2 + 2)])

        H = 2*x**2 + 3*y**3
        self.assertEqual(newton_data(H),
                         [(3, 2, 6, 3*x + 2)])

        H = 2*x**2 + 3*y**4
        self.assertEqual(newton_data(H),
                         [(2, 1, 4, 3*x**2 + 2)])

    def test_general_segment(self):
        H = x**2 + y
        self.assertEqual(newton_data(H),
                         [(1, 1, 1, x)])

        H = x**3 + y**2
        self.assertEqual(newton_data(H),
                         [(1, 1, 2, x**2)])

        H = x**5 + y**3
        self.assertEqual(newton_data(H),
                         [(1, 1, 3, x**3)])

    def test_colinear(self):
        H = 2*x**4 + 3*x**2*y**2 + 5*y**4
        self.assertEqual(newton_data(H),
                         [(1, 1, 4, 5*x**4 + 3*x**2 + 2)])

        H = 2*x**4 + 3*x**3*y + 5*x**2*y**2 + 7*y**4
        self.assertEqual(newton_data(H),
                         [(1, 1, 4, 7*x**4 + 5*x**2 + 3*x + 2)])

        H = 2*x**4 + 3*x**2*y**2 + 5*x*y**3 + 7*y**4
        self.assertEqual(newton_data(H),
                         [(1, 1, 4, 7*x**4 + 5*x**3 + 3*x**2 + 2)])

    def test_multiple(self):
        H = 2*x**2 + 3*x*y + 5*y**3
        self.assertEqual(newton_data(H),
                         [(1, 1, 2, 3*x + 2),
                          (2, 1, 3, 5*x + 3)])

    def test_general_to_colinear(self):
        H = 2*x**5 + 3*x**2*y**2 + 5*y**4
        self.assertEqual(newton_data(H),
                         [(1, 1, 4, 5*x**4 + 3*x**2)])

        H = 2*x**5 + 3*x**3*y + 5*x**2*y**2 + 7*y**4
        self.assertEqual(newton_data(H),
                         [(1, 1, 4, 7*x**4 + 5*x**2 + 3*x)])

        H = 2*x**5 + 3*x**3*y + 5*x**2*y**2 + 7*y**6
        self.assertEqual(newton_data(H),
                         [(1, 1, 4, 5*x**2 + 3*x),
                          (2, 1, 6, 7*x**2 + 5)])


class TestNewPolynomial(unittest.TestCase):

    def setUp(self):
        self.p = R.random_element(degree=5)
        self.q = R.random_element(degree=5)

    def test_null_transform(self):
        H = self.p*self.q
        q,m,l,xi = 1,0,0,0
        Hprime = transform_newton_polynomial(H,q,m,l,xi)
        Htest = H
        self.assertEqual(Hprime, Htest)

    def test_yshift(self):
        H = self.p*self.q
        q,m,l,xi = 1,0,0,1
        Hprime = transform_newton_polynomial(H,q,m,l,xi)
        Htest = H(y=1+y)
        self.assertEqual(Hprime, Htest)

    def test_y(self):
        H = y**2
        q,m,l,xi = 0,1,0,1
        Hprime = transform_newton_polynomial(H,q,m,l,xi)
        Htest = x**2*(1 + y)**2
        self.assertEqual(Hprime, Htest)

        q,m,l,xi = 0,1,2,1
        Hprime = transform_newton_polynomial(H,q,m,l,xi)
        Htest = (1 + y)**2
        self.assertEqual(Hprime, Htest)


class TestNewtonIteration(unittest.TestCase):

    def test_trivial(self):
        G = y - x
        S = newton_iteration(G,3)
        self.assertEqual(S,x)

        G = y - x**2
        S = newton_iteration(G,3)
        self.assertEqual(S,x**2)

    def test_sqrt(self):
        # recenter sqrt(x) at x+1
        G = (y+1)**2 - (x+1)
        S = newton_iteration(G,9).truncate(x,10) + 1

        z = var('z')
        series = taylor(sqrt(z),z,1,9)
        series = R(series.subs({z:x+1}))
        self.assertEqual(S,series)

    def test_cuberoot(self):
        # recenter cuberoot(x) at x+1
        G = (y+1)**3 - (x+1)
        S = newton_iteration(G,9).truncate(x,10) + 1

        z = var('z')
        series = taylor(z**(QQ(1)/QQ(3)),z,1,9)
        series = R(series.subs({z:x+1}))
        self.assertEqual(S,series)

    def test_geometric(self):
        G = (1-x)*y - 1
        S = newton_iteration(G,9).truncate(x,10)

        z = var('z')
        series = taylor(1/(1-z),z,0,9)
        series = R(series.subs({z:x}))
        self.assertEqual(S,series)

    def test_n(self):
        # test if the solution is indeed given to the desired terms
        G = y - x**2
        S = newton_iteration(G,0)
        self.assertEqual(S,0)

        G = y - x**2
        S = newton_iteration(G,3)
        self.assertEqual(S,x**2)


class TestPuiseuxRational(AbelfunctionsTestCase):
    def is_G_vanishing(self, fmonic):
        # each G should satisfy G(0,0) = 0 and G_y(0,0) = 0
        for G,P,Q in puiseux_rational(fmonic):
            self.assertEqual(G(x=0,y=0),0)
    def test_G_vanishing2(self):
        self.is_G_vanishing(self.f2)
    def test_G_vanishing4(self):
        self.is_G_vanishing(self.f4)
    def test_G_vanishing5(self):
        self.is_G_vanishing(self.f5)
    def test_G_vanishing6(self):
        self.is_G_vanishing(self.f6)
    def test_G_vanishing7(self):
        self.is_G_vanishing(self.f7)
    def test_G_vanishing9(self):
        self.is_G_vanishing(self.f9)


class TestAlmostMonicize(AbelfunctionsTestCase):
    def test_monic(self):
        f = y**2 + x
        g,transform = almost_monicize(f)
        self.assertEqual(f,g)

    def test_partially_monic(self):
        g,transform = almost_monicize(self.f1)
        self.assertEqual(self.f1,g)

        f = (x**2 + 1)*y
        g,transform = almost_monicize(f)
        self.assertEqual(f,g)

    def test_not_monic_simple(self):
        f = x**2*y
        g,transform = almost_monicize(f)
        self.assertEqual(g,y)
        self.assertEqual(transform,x**2)

        f = x**2*y + 1
        g,transform = almost_monicize(f)
        self.assertEqual(g,y + 1)
        self.assertEqual(transform,x**2)

        f = x**2*y + x
        g,transform = almost_monicize(f)
        self.assertEqual(g,y + x)
        self.assertEqual(transform,x**2)

        f = x*y**2 + y + x
        g,transform = almost_monicize(f)
        self.assertEqual(g,y**2 + y + x**2)
        self.assertEqual(transform,x)

        f = x**3*y**2 + y + x
        g,transform = almost_monicize(f)
        self.assertEqual(g,y**2 + y + x**4)
        self.assertEqual(transform,x**3)

    def test_not_monic(self):
        f = x**7*y**3 + 2*y - x**7
        g,transform = almost_monicize(f)

        self.assertEqual(g,y**3 + 2*x*y - x**12)
        self.assertEqual(transform,x**4)

        g,transform = almost_monicize(self.f8)
        self.assertEqual(g,-x**4 + 2*x**2*y**5 + y**6)
        self.assertEqual(transform,x)

    def test_issue70(self):
        f = -x**5 + x*y**4 + y**2
        g, transform = almost_monicize(f)
        self.assertEqual(g,y**4 + y**2*x - x**8)
        self.assertEqual(transform,x)


class TestPuiseux(AbelfunctionsTestCase):
    def setUp(self):
        self.f22 = y**3 - x**5
        self.f23 = (y - 1 - 2*x - x**2)*(y - 1 - 2*x - x**7)
        self.f27 = (y**2 - 2*x**3)*(y**2-2*x**2)*(y**3-2*x)
        super(TestPuiseux,self).setUp()

    def get_PQ(self,f, a=0):
        p = puiseux(f,a)
        if p:
            series = [(P._xpart,P._ypart) for P in p]
        else:
            series = []
        return series

    def test_PQ_f1(self):
        series = self.get_PQ(self.f1)
        x,y = self.f1.parent().gens()
        x = QQbar['x,y'](x)
        y = QQbar['x,y'](y)
        six.assertCountEqual(self, 
            series,
            [(x**2, x**4*(x*(y + 1) + 1))])

    def test_PQ_f2(self):
        series = self.get_PQ(self.f2)
        x,y = self.f2.parent().gens()
        x = QQbar['x,y'](x)
        y = QQbar['x,y'](y)
        six.assertCountEqual(self, 
            series,
            [(x, x**2*y),
             (-x**2/2, -x**3*(y + 1)/2)])

    def test_PQ_f2_oo(self):
        series = self.get_PQ(self.f2, a=infinity)
        x,y = self.f2.parent().gens()
        x = QQbar['x,y'](x)
        y = QQbar['x,y'](y)
        six.assertCountEqual(self, 
            series,
            [(1/x**3, (x**2*y + x**2)/x**9)])

    def test_PQ_f3(self):
        # awaiting RootOf simplification issues
        pass

    def test_PQ_f4(self):
        series = self.get_PQ(self.f4)
        x,y = self.f4.parent().gens()
        x = QQbar['x,y'](x)
        y = QQbar['x,y'](y)
        six.assertCountEqual(self, 
            series,
            [(x, x*(y + 1)),
             (x, x*(y - 1))])

    def test_PQ_f4_oo(self):
        series = self.get_PQ(self.f4, a=infinity)
        x,y = self.f4.parent().gens()
        x = QQbar['x,y'](x)
        y = QQbar['x,y'](y)
        six.assertCountEqual(self, 
            series,
            [(1/(-x**2), (x*y + x)/x**4)])

    def test_PQ_f7(self):
        S = QQ['t']
        t = S.gen()
        r0,r1,r2 = (t**3 - t**2 + 1).roots(QQbar, multiplicities=False)

        series = self.get_PQ(self.f7)
        x,y = self.f7.parent().gens()
        x = QQbar['x,y'](x)
        y = QQbar['x,y'](y)
        six.assertCountEqual(self, 
            series,
            [(x, y + r0),
             (x, y + r1),
             (x, y + r2)])

    def test_PQ_f22(self):
        series = self.get_PQ(self.f22)
        x,y = self.f22.parent().gens()
        x = QQbar['x,y'](x)
        y = QQbar['x,y'](y)
        six.assertCountEqual(self, 
            series,
            [(x**3, x**5*(y + 1))])

    def test_PQ_f22_oo(self):
        series = self.get_PQ(self.f22, a=infinity)
        x,y = self.f2.parent().gens()
        x = QQbar['x,y'](x)
        y = QQbar['x,y'](y)
        six.assertCountEqual(self, 
            series,
            [(1/x**3, (x*y + x)/x**6)])

    def test_PQ_f23(self):
        series = self.get_PQ(self.f23)
        x,y = self.f23.parent().gens()
        x = QQbar['x,y'](x)
        y = QQbar['x,y'](y)
        six.assertCountEqual(self, 
            series,
            [(x, x*(x*y + 2) + 1),
             (x, x*(x*(y + 1) + 2) + 1)])

    def test_PQ_f23_oo(self):
        series = self.get_PQ(self.f23, a=infinity)
        x,y = self.f23.parent().gens()
        x = QQbar['x,y'](x)
        y = QQbar['x,y'](y)
        six.assertCountEqual(self, 
            series,
            [(1/x,y/x**7), (1/x,(1+y)/x**7)])

    def test_PQ_f27(self):
        S = QQ['t']
        t = S.gen()
        sqrt2 = (t**2 - 2).roots(QQbar, multiplicities=False)[0]

        series = self.get_PQ(self.f27)
        x,y = self.f27.parent().gens()
        x = QQbar['x,y'](x)
        y = QQbar['x,y'](y)
        six.assertCountEqual(self, 
            series,
            [(x, x*(y + sqrt2)),
             (x, x*(y - sqrt2)),
             (x**2/2, x**3*(y + 1)/2),
             (x**3/2, x*(y + 1))])

    def test_hyperelliptic_oo(self):
        R = QQ['x,y']
        x,y = R.gens()
        f = y**2 - (x**2 - 9)*(x**2 - 4)*(x**2 - 1)
        series = self.get_PQ(f,a=infinity)
        x = QQbar['x,y'](x)
        y = QQbar['x,y'](y)
        six.assertCountEqual(self, 
            series,
            [(1/x, (y+1)/x**3),
             (1/x, (y-1)/x**3)])

class TestEvaluation(AbelfunctionsTestCase):

    def test_extend_to_t(self):
        p = puiseux(self.f2,0)
        pi = p[0]
        ti = 0.1

        # 1e-8
        pi.extend_to_t(ti, 1e-8)
        xt = pi.eval_x(ti)
        yt = pi.eval_y(ti)
        error = abs(self.f2(xt,yt))
        self.assertLess(error, 1e-8)

        # 1e-14
        pi.extend_to_t(ti, curve_tol=1e-14)
        xt = pi.eval_x(ti)
        yt = pi.eval_y(ti)
        error = abs(self.f2(xt,yt))
        self.assertLess(error, 1e-14)

        # 1e-20 (multi-precise)
        pi.extend_to_t(ti, curve_tol=1e-20)
        xt = pi.eval_x(ti)
        yt = pi.eval_y(ti)
        error = abs(self.f2(xt,yt))
        self.assertLess(error, 1e-20)

    def test_extend_to_t_oo(self):
        p = puiseux(self.f2,'oo')
        pi = p[0]
        ti = 100

        # 1e-8
        pi.extend_to_t(ti, 1e-8)
        xt = pi.eval_x(ti)
        yt = pi.eval_y(ti)
        error = abs(self.f2(xt,yt))
        self.assertLess(error, 1e-8)

        # 1e-12
        pi.extend_to_t(ti, curve_tol=1e-12)
        xt = pi.eval_x(ti)
        yt = pi.eval_y(ti)
        error = abs(self.f2(xt,yt))
        self.assertLess(error, 1e-12)
