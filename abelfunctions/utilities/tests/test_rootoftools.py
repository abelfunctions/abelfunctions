import abelfunctions
import unittest

from abelfunctions.utilities.rootoftools import all_roots, rootofsimp
from sympy import RootOf, expand, sin
from sympy.abc import x,y,z,t


class TestAllRoots(unittest.TestCase):
    r"""General tests for all_roots."""

    def test_dummy_variable(self):
        # tests that the defining polynomial of the ouput roots uses a dummy
        # variable based on the input
        r = RootOf(y**2 + 1, 0, radicals=False)
        f = x**2 + r
        roots = all_roots(f, x)
        dummy_var = '_x'
        for root in roots:
            self.assertTrue(str(root.poly.gen) == dummy_var)


class TestErrors(unittest.TestCase):
    r"""Test if errors are raised in appropriate situations."""

    def test_non_RootOf(self):
        r = RootOf(y**2 + 1, 0, radicals=True)
        f = x + r
        with self.assertRaises(NotImplementedError):
            roots = all_roots(f, x)

        r = RootOf(y**2 - 2, 0, radicals=True)
        f = x**2 + r
        with self.assertRaises(NotImplementedError):
            roots = all_roots(f, x)

    def test_multiple_RootOf(self):
        r = RootOf(y**5 + y + 1, 0, radicals=False)
        s = RootOf(z**8 + z**7 + 2*z**3 - 3, 0, radicals=False)
        f = x**2 + r*x + s
        with self.assertRaises(NotImplementedError):
            roots = all_roots(f, x)


class TestDegree(unittest.TestCase):
    r"""Tests if the correct number of roots are returned."""

    def test_trivial(self):
        f = x + 1
        roots = all_roots(f, x)
        self.assertTrue(len(roots) == f.as_poly(x).degree())

        f = x**2 + 1
        roots = all_roots(f, x)
        self.assertTrue(len(roots) == f.as_poly(x).degree())

        f = x**3 + 4*x**2 + 1
        roots = all_roots(f, x)
        self.assertTrue(len(roots) == f.as_poly(x).degree())

    def test_simple(self):
        r = RootOf(y**2 + 1, 0, radicals=False)
        f = x + r
        roots = all_roots(f, x)
        self.assertTrue(len(roots) == f.as_poly(x).degree())

        f = x**2 + r
        roots = all_roots(f, x)
        self.assertTrue(len(roots) == f.as_poly(x).degree())

        f = x**3 + r*x**2 + 1
        roots = all_roots(f, x)
        self.assertTrue(len(roots) == f.as_poly(x).degree())

    def test_degree(self):
        r = RootOf(y**5 - 2*y + 1, 0, radicals=False)
        f = 3*r*x**2 - 2*x + r**3
        roots = all_roots(f, x)
        self.assertTrue(len(roots) == f.as_poly(x).degree())


class TestIsZero(unittest.TestCase):
    r"""Tests if the roots returned are indeed roots of the input polynomial"""

    def all_is_zero(self, f, gen, roots, tol=1e-15):
        r"""Returns `True` if f is near zero at each root."""
        f = f.as_poly(gen)
        is_zero = map(lambda root: abs(f.evalf(subs={gen:root})) < tol, roots)
        return all(is_zero)

    def test_trivial(self):
        f = 1
        roots = all_roots(f, x)
        self.assertTrue(roots == [])

        f = x + 1
        roots = all_roots(f, x)
        self.assertTrue(self.all_is_zero(f, x, roots))

        f = x**2 + 1
        roots = all_roots(f, x)
        self.assertTrue(self.all_is_zero(f, x, roots))

        f = x**3 + 4*x**2 + 1
        roots = all_roots(f, x)
        self.assertTrue(self.all_is_zero(f, x, roots))

    def test_simple(self):
        r = RootOf(y**2 + 1, 0, radicals=False)

        f = r
        roots = all_roots(f, x)
        self.assertTrue(roots == [])

        f = x + r
        roots = all_roots(f, x)
        self.assertTrue(self.all_is_zero(f, x, roots))

        f = x**2 + r
        roots = all_roots(f, x)
        self.assertTrue(self.all_is_zero(f, x, roots))

        f = x**3 + r*x**2 + 1
        roots = all_roots(f, x)
        self.assertTrue(self.all_is_zero(f, x, roots))

    def test_is_zero(self):
        r = RootOf(y**5 - 2*y + 1, 0, radicals=False)
        f = 3*r*x**2 - 2*x + r**3
        roots = all_roots(f, x)
        self.assertTrue(self.all_is_zero(f, x, roots))



class TestRootOfSimp(unittest.TestCase):
    r"""Testing simplification of RootOf expressions using `rootofsimp()`."""

    def test_rootofsimp(self):
        r = RootOf(x**5 - 5*x + 12, 0)
        f, g = r**5, 5*r - 12
        self.assertTrue(f != g and rootofsimp(f) == g)

        f, g = r**10, 25*r**2 - 120*r + 144
        self.assertTrue(f != g and rootofsimp(f) == g)

        f, g = r**4*(1 + r), r**4 + 5*r - 12
        self.assertTrue(f != g and rootofsimp(f) == g)

    def test_modular_inversion(self):
        r = RootOf(x**5 - 5*x + 12, 0)
        f, g = 1/r, (-r**4 + 5)/12
        self.assertTrue(f != g and rootofsimp(f) == g)

        f, g = (-r**4 + 5)/12, r
        self.assertTrue(f*g != 1 and rootofsimp(f*g) == 1)

    def test_multiple_roots(self):
        r = RootOf(x**5 - 5*x + 12, 0)
        s = RootOf(x**4 - x**2 + 1, 0)
        t = RootOf(x**4 - x**2 + 1, 1)

        f, g = r**5 * s**4, expand((5*r - 12)*(s**2 - 1))
        self.assertTrue(f != g and rootofsimp(f).expand() == g)

        # test multiple roots of the same polynomial
        r0 = RootOf(x**5 - 5*x + 12, 3)
        r1 = RootOf(x**5 - 5*x + 12, 4)
        f, g = r0**5 - 5*r1 + 12, 5*r0 - 5*r1
        self.assertTrue(f != g and rootofsimp(f) == g)

        f, g = (s**4 + 1)*(t**4 + 1), s**2*t**2
        self.assertTrue(f != g and rootofsimp(f) == g)

    def test_forced_radicals(self):
        # test RootOfs with forced radicals=False
        u = RootOf(x**2 + 1, 0, radicals=False)
        v = RootOf(x**2 - 2, 0, radicals=False)

        f, g = u**2, -1
        self.assertTrue(f != g and rootofsimp(f) == g)

        f, g = u**3, -u
        self.assertTrue(f != g and rootofsimp(f) == g)

        f, g = v**2, 2
        self.assertTrue(f != g and rootofsimp(f) == g)

        f, g = u**2*v**2, -2
        self.assertTrue(f != g and rootofsimp(f) == g)

        # test no-op when expression is not polynomial the RootOf
        f, g = sin(u**2 + 1), 0
        self.assertTrue(f != g and rootofsimp(f) != g)

    def test_notinvertible(self):
        r = RootOf(x**2 + 1, 0, radicals=False)
        expr = 1/(r**2 + 1)
        with self.assertRaises(ZeroDivisionError):
            rootofsimp(expr)
