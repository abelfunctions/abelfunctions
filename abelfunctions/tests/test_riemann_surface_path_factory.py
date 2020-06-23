import unittest
import six

from abelfunctions.tests.test_abelfunctions import AbelfunctionsTestCase

from abelfunctions.complex_path import (
    ComplexPathPrimitive,
    ComplexPath,
    ComplexLine,
    ComplexArc,
    ComplexRay,
)
from abelfunctions.riemann_surface import RiemannSurface
from abelfunctions.riemann_surface_path import (
    RiemannSurfacePath,
    RiemannSurfacePathPuiseux,
    RiemannSurfacePathSmale
    )
from abelfunctions.riemann_surface_path_factory import RiemannSurfacePathFactory
from abelfunctions.utilities import Permutation

from numpy import sqrt, pi
from sage.all import QQ, QQbar, infinity, I

class TestConstruction(unittest.TestCase):
    def setUp(self):
        R = QQ['x,y']
        x,y = R.gens()

        f1 = y**2 - x
        self.f1 = f1
        self.X1 = RiemannSurface(f1)

        f2 = y**2 - (x**2 + 1)
        self.f2 = f2
        self.X2 = RiemannSurface(f2)

        f3 = y**2 - (x**4 - 1)
        self.f3 = f3
        self.X3 = RiemannSurface(f3)

    def test_construction(self):
        _ = RiemannSurfacePathFactory(self.X1)
        _ = RiemannSurfacePathFactory(self.X2)
        _ = RiemannSurfacePathFactory(self.X3)

    def test_base_point_and_sheets(self):
        PF = RiemannSurfacePathFactory(self.X1)
        self.assertAlmostEqual(PF.base_point, -1)
        sheets = PF.base_sheets
        self.assertAlmostEqual(sheets[0], -1.j)
        self.assertAlmostEqual(sheets[1], 1.j)

        PF = RiemannSurfacePathFactory(self.X2)
        self.assertAlmostEqual(PF.base_point, -1)
        sheets = PF.base_sheets
        self.assertAlmostEqual(sheets[0], -sqrt(2))
        self.assertAlmostEqual(sheets[1], sqrt(2))

        PF = RiemannSurfacePathFactory(self.X3)
        self.assertAlmostEqual(PF.base_point, -2)
        sheets = PF.base_sheets
        self.assertAlmostEqual(sheets[0], -sqrt(15))
        self.assertAlmostEqual(sheets[1], sqrt(15))

    def test_base_point_and_sheets_given(self):
        PF = RiemannSurfacePathFactory(self.X1, base_point=-2)
        self.assertAlmostEqual(PF.base_point, -2)
        sheets = PF.base_sheets
        self.assertAlmostEqual(sheets[0], -1.j*sqrt(2))
        self.assertAlmostEqual(sheets[1], 1.j*sqrt(2))

        PF = RiemannSurfacePathFactory(self.X1, base_sheets=[1.j, -1.j])
        self.assertAlmostEqual(PF.base_point, -1)
        sheets = PF.base_sheets
        self.assertAlmostEqual(sheets[0], 1.j)
        self.assertAlmostEqual(sheets[1], -1.j)

    def test_incorrect_sheets_above_base_point(self):
        with self.assertRaises(ValueError):
            PF = RiemannSurfacePathFactory(self.X1, base_point=-1,
                                           base_sheets=[42, 101])

class TestFromComplexPath(unittest.TestCase):
    def setUp(self):
        R = QQ['x,y']
        x,y = R.gens()

        f1 = y**2 - x
        self.f1 = f1
        self.X1 = RiemannSurface(f1)

        f2 = y**2 - (x**2 + 1)
        self.f2 = f2
        self.X2 = RiemannSurface(f2)

        f3 = y**2 - (x**4 - 1)
        self.f3 = f3
        self.X3 = RiemannSurface(f3)

    def test_primitive_line_smale(self):
        PF = RiemannSurfacePathFactory(self.X1, base_point=-1,
                                       base_sheets=[-1.j,1.j])
        gamma_x = ComplexLine(-1,-2)
        gamma = PF.RiemannSurfacePath_from_complex_path(gamma_x)
        self.assertAlmostEqual(gamma.get_x(1.0), -2)
        self.assertAlmostEqual(gamma.get_y(1.0)[0], -1.j*sqrt(2))
        self.assertAlmostEqual(gamma.get_y(1.0)[1], 1.j*sqrt(2))

        # swap the base sheets
        PF = RiemannSurfacePathFactory(self.X1, base_point=-1,
                                       base_sheets=[1.j,-1.j])
        gamma_x = ComplexLine(-1,-2)
        gamma = PF.RiemannSurfacePath_from_complex_path(gamma_x)
        self.assertAlmostEqual(gamma.get_x(1.0), -2)
        self.assertAlmostEqual(gamma.get_y(1.0)[0], 1.j*sqrt(2))
        self.assertAlmostEqual(gamma.get_y(1.0)[1], -1.j*sqrt(2))

    def test_primitive_arc_smale(self):
        PF = RiemannSurfacePathFactory(self.X1, base_point=-1,
                                       base_sheets=[-1.j,1.j])
        gamma_x = ComplexArc(1, 0, pi, pi)
        gamma = PF.RiemannSurfacePath_from_complex_path(gamma_x)
        self.assertAlmostEqual(gamma.get_x(1.0), 1)
        self.assertAlmostEqual(gamma.get_y(1.0)[0], 1)
        self.assertAlmostEqual(gamma.get_y(1.0)[1], -1)

        # swap the base sheets
        PF = RiemannSurfacePathFactory(self.X1, base_point=-1,
                                       base_sheets=[1.j,-1.j])
        gamma_x = ComplexArc(1, 0, pi, pi)
        gamma = PF.RiemannSurfacePath_from_complex_path(gamma_x)
        self.assertAlmostEqual(gamma.get_x(1.0), 1)
        self.assertAlmostEqual(gamma.get_y(1.0)[0], -1)
        self.assertAlmostEqual(gamma.get_y(1.0)[1], 1)

    def test_primitive_line_puiseux(self):
        PF = RiemannSurfacePathFactory(self.X1, base_point=-1,
                                       base_sheets=[-1.j,1.j])
        gamma_x = ComplexLine(-1,0)
        gamma = PF.RiemannSurfacePath_from_complex_path(gamma_x)
        self.assertAlmostEqual(gamma.get_x(1.0), 0)
        self.assertAlmostEqual(gamma.get_y(1.0)[0], 0)
        self.assertAlmostEqual(gamma.get_y(1.0)[1], 0)

        self.assertAlmostEqual(gamma.get_y(0.5)[0], -1.j*sqrt(0.5))
        self.assertAlmostEqual(gamma.get_y(0.5)[1], 1.j*sqrt(0.5))

        # swap the base sheets
        PF = RiemannSurfacePathFactory(self.X1, base_point=-1,
                                       base_sheets=[1.j,-1.j])
        gamma_x = ComplexLine(-1,0)
        gamma = PF.RiemannSurfacePath_from_complex_path(gamma_x)
        self.assertAlmostEqual(gamma.get_x(1.0), 0)
        self.assertAlmostEqual(gamma.get_y(1.0)[0], 0)
        self.assertAlmostEqual(gamma.get_y(1.0)[1], 0)

        self.assertAlmostEqual(gamma.get_y(0.5)[0], 1.j*sqrt(0.5))
        self.assertAlmostEqual(gamma.get_y(0.5)[1], -1.j*sqrt(0.5))

    def test_composite_line_smale(self):
        PF = RiemannSurfacePathFactory(self.X1, base_point=-3,
                                       base_sheets=[-1.j*sqrt(3),1.j*sqrt(3)])
        gamma_x = ComplexLine(-3,-2) + ComplexLine(-2,-1)
        gamma = PF.RiemannSurfacePath_from_complex_path(gamma_x)
        self.assertAlmostEqual(gamma.get_x(0.0), -3)
        self.assertAlmostEqual(gamma.get_y(0.0)[0], -1.j*sqrt(3))
        self.assertAlmostEqual(gamma.get_y(0.0)[1], 1.j*sqrt(3))

        self.assertAlmostEqual(gamma.get_x(0.5), -2)
        self.assertAlmostEqual(gamma.get_y(0.5)[0], -1.j*sqrt(2))
        self.assertAlmostEqual(gamma.get_y(0.5)[1], 1.j*sqrt(2))

        self.assertAlmostEqual(gamma.get_x(1.0), -1)
        self.assertAlmostEqual(gamma.get_y(1.0)[0], -1.j)
        self.assertAlmostEqual(gamma.get_y(1.0)[1], 1.j)

        # swap the base sheets
        PF = RiemannSurfacePathFactory(self.X1, base_point=-3,
                                       base_sheets=[1.j*sqrt(3),-1.j*sqrt(3)])
        gamma_x = ComplexLine(-3,-2) + ComplexLine(-2,-1)
        gamma = PF.RiemannSurfacePath_from_complex_path(gamma_x)
        self.assertAlmostEqual(gamma.get_x(0.0), -3)
        self.assertAlmostEqual(gamma.get_y(0.0)[0], 1.j*sqrt(3))
        self.assertAlmostEqual(gamma.get_y(0.0)[1], -1.j*sqrt(3))

        self.assertAlmostEqual(gamma.get_x(0.5), -2)
        self.assertAlmostEqual(gamma.get_y(0.5)[0], 1.j*sqrt(2))
        self.assertAlmostEqual(gamma.get_y(0.5)[1], -1.j*sqrt(2))

        self.assertAlmostEqual(gamma.get_x(1.0), -1)
        self.assertAlmostEqual(gamma.get_y(1.0)[0], 1.j)
        self.assertAlmostEqual(gamma.get_y(1.0)[1], -1.j)

    def test_composite_line_mixed(self):
        PF = RiemannSurfacePathFactory(self.X1, base_point=-2,
                                       base_sheets=[-1.j*sqrt(2),1.j*sqrt(2)])
        gamma_x = ComplexLine(-2,-1) + ComplexLine(-1,0)
        gamma = PF.RiemannSurfacePath_from_complex_path(gamma_x)
        self.assertEqual(len(gamma.segments), 2)
        self.assertTrue(
            isinstance(gamma.segments[0], RiemannSurfacePathSmale))
        self.assertTrue(
            isinstance(gamma.segments[1], RiemannSurfacePathPuiseux))

        self.assertAlmostEqual(gamma.get_x(0.0), -2)
        self.assertAlmostEqual(gamma.get_y(0.0)[0], -1.j*sqrt(2))
        self.assertAlmostEqual(gamma.get_y(0.0)[1], 1.j*sqrt(2))

        self.assertAlmostEqual(gamma.get_x(0.5), -1)
        self.assertAlmostEqual(gamma.get_y(0.5)[0], -1.j)
        self.assertAlmostEqual(gamma.get_y(0.5)[1], 1.j)

        self.assertAlmostEqual(gamma.get_x(1.0), 0)
        self.assertAlmostEqual(gamma.get_y(1.0)[0], 0)
        self.assertAlmostEqual(gamma.get_y(1.0)[1], 0)

        # swap the base sheets
        PF = RiemannSurfacePathFactory(self.X1, base_point=-2,
                                       base_sheets=[1.j*sqrt(2),-1.j*sqrt(2)])
        gamma_x = ComplexLine(-2,-1) + ComplexLine(-1,0)
        gamma = PF.RiemannSurfacePath_from_complex_path(gamma_x)
        self.assertEqual(len(gamma.segments), 2)
        self.assertTrue(
            isinstance(gamma.segments[0], RiemannSurfacePathSmale))
        self.assertTrue(
            isinstance(gamma.segments[1], RiemannSurfacePathPuiseux))

        self.assertAlmostEqual(gamma.get_x(0.0), -2)
        self.assertAlmostEqual(gamma.get_y(0.0)[0], 1.j*sqrt(2))
        self.assertAlmostEqual(gamma.get_y(0.0)[1], -1.j*sqrt(2))

        self.assertAlmostEqual(gamma.get_x(0.5), -1)
        self.assertAlmostEqual(gamma.get_y(0.5)[0], 1.j)
        self.assertAlmostEqual(gamma.get_y(0.5)[1], -1.j)

        self.assertAlmostEqual(gamma.get_x(1.0), 0)
        self.assertAlmostEqual(gamma.get_y(1.0)[0], 0)
        self.assertAlmostEqual(gamma.get_y(1.0)[1], 0)

class TestMonodromyPath(unittest.TestCase):
    def setUp(self):
        R = QQ['x,y']
        x,y = R.gens()

        f1 = y**2 - x
        self.f1 = f1
        self.X1 = RiemannSurface(f1)

        f2 = y**2 - (x**2 + 1)
        self.f2 = f2
        self.X2 = RiemannSurface(f2)

        f3 = y**2 - (x**4 - 1)
        self.f3 = f3
        self.X3 = RiemannSurface(f3)

        f4 = y**3 - (x**2 + 1)
        self.f4 = f4
        self.X4 = RiemannSurface(f4)

    def test_monodromy_path(self):
        PF = RiemannSurfacePathFactory(self.X1, base_point=-2,
                                       base_sheets=[-1.j*sqrt(2),1.j*sqrt(2)])
        gamma_x = PF.complex_path_factory.monodromy_path(0)
        gamma = PF.RiemannSurfacePath_from_complex_path(gamma_x)
        radius = PF.complex_path_factory.radius(0)
        self.assertAlmostEqual(gamma.get_x(0.0), -2)
        self.assertAlmostEqual(gamma.get_x(0.5), radius)
        self.assertAlmostEqual(gamma.get_x(1.0), -2)

        self.assertAlmostEqual(gamma.get_y(0.0)[0], -1.j*sqrt(2))
        self.assertAlmostEqual(gamma.get_y(0.0)[1], 1.j*sqrt(2))
        self.assertAlmostEqual(gamma.get_y(1.0)[0], 1.j*sqrt(2))
        self.assertAlmostEqual(gamma.get_y(1.0)[1], -1.j*sqrt(2))

        PF = RiemannSurfacePathFactory(self.X2, base_point=-2,
                                       base_sheets=[-sqrt(5),sqrt(5)])
        gamma_x = PF.complex_path_factory.monodromy_path(-I)
        gamma = PF.RiemannSurfacePath_from_complex_path(gamma_x)
        self.assertAlmostEqual(gamma.get_x(0.0), -2)
        self.assertAlmostEqual(gamma.get_x(1.0), -2)

        self.assertAlmostEqual(gamma.get_y(0.0)[0], -sqrt(5))
        self.assertAlmostEqual(gamma.get_y(0.0)[1], sqrt(5))
        self.assertAlmostEqual(gamma.get_y(1.0)[0], sqrt(5))
        self.assertAlmostEqual(gamma.get_y(1.0)[1], -sqrt(5))

    def test_monodromy(self):
        oo = infinity

        PF = RiemannSurfacePathFactory(self.X1)
        branch_points, permutations = PF.monodromy_group()
        six.assertCountEqual(self, branch_points, [QQbar(0), oo])
        self.assertEqual(permutations[0], Permutation([1,0]))

        PF = RiemannSurfacePathFactory(self.X2)
        branch_points, permutations = PF.monodromy_group()
        six.assertCountEqual(self, branch_points,
                              [QQbar(z) for z in [-I, I]])
        self.assertEqual(permutations[0], Permutation([1,0]))
        self.assertEqual(permutations[1], Permutation([1,0]))

        PF = RiemannSurfacePathFactory(self.X3)
        branch_points, permutations = PF.monodromy_group()
        six.assertCountEqual(self, branch_points,
                              [QQbar(z) for z in [-I, -1, 1, I]])
        self.assertEqual(permutations[0], Permutation([1,0]))
        self.assertEqual(permutations[1], Permutation([1,0]))
        self.assertEqual(permutations[2], Permutation([1,0]))
        self.assertEqual(permutations[3], Permutation([1,0]))

        PF = RiemannSurfacePathFactory(self.X4)
        branch_points, permutations = PF.monodromy_group()
        six.assertCountEqual(self, branch_points,
                              [QQbar(z) for z in [-I, I]] + [oo])
        self.assertEqual(permutations[0], Permutation([2,0,1]))
        self.assertEqual(permutations[1], Permutation([2,0,1]))
        self.assertEqual(permutations[2], Permutation([2,0,1]))

    def test_monodromy_path_rotations(self):
        # test that if we rotate around the branch point a number of times
        # equal to the order of the permutation we arrive where we started.

        # all branch points have order two permutations
        PF = RiemannSurfacePathFactory(self.X1)
        gamma = PF.monodromy_path(0, nrots=2)
        y0 = gamma.get_y(0.0)
        y1 = gamma.get_y(1.0)
        self.assertAlmostEqual(y0[0], y1[0])
        self.assertAlmostEqual(y0[1], y1[1])

        PF = RiemannSurfacePathFactory(self.X2)
        gamma = PF.monodromy_path(QQbar(-I), nrots=2)
        y0 = gamma.get_y(0.0)
        y1 = gamma.get_y(1.0)
        self.assertAlmostEqual(y0[0], y1[0])
        self.assertAlmostEqual(y0[1], y1[1])

        # all branch points have order three permutations
        PF = RiemannSurfacePathFactory(self.X4)
        gamma = PF.monodromy_path(QQbar(I), nrots=3)
        y0 = gamma.get_y(0.0)
        y1 = gamma.get_y(1.0)
        self.assertAlmostEqual(y0[0], y1[0])
        self.assertAlmostEqual(y0[1], y1[1])
        self.assertAlmostEqual(y0[2], y1[2])


class TestCycles(unittest.TestCase):
    def setUp(self):
        R = QQ['x,y']
        x,y = R.gens()

        f1 = y**2 - x
        self.f1 = f1
        self.X1 = RiemannSurface(f1)

        f2 = y**2 - (x**2 + 1)
        self.f2 = f2
        self.X2 = RiemannSurface(f2)

        f3 = y**2 - (x**4 - 1)
        self.f3 = f3
        self.X3 = RiemannSurface(f3)

        f4 = y**3 - (x**2 + 1)
        self.f4 = f4
        self.X4 = RiemannSurface(f4)

    def test_closed(self):
        # tests that the cycles are, indeed, cycles
        PF = RiemannSurfacePathFactory(self.X3)
        cycles = PF.a_cycles()
        for a in cycles:
            x0 = a.get_x(0)
            y0 = a.get_y(0)
            x1 = a.get_x(1)
            y1 = a.get_y(1)
            self.assertAlmostEqual(x0, x1)
            self.assertAlmostEqual(y0[0], y1[0])
            self.assertAlmostEqual(y0[1], y1[1])

        PF = RiemannSurfacePathFactory(self.X4)
        cycles = PF.a_cycles()
        for a in cycles:
            x0 = a.get_x(0)
            y0 = a.get_y(0)
            x1 = a.get_x(1)
            y1 = a.get_y(1)
            self.assertAlmostEqual(x0, x1)
            self.assertAlmostEqual(y0[0], y1[0])
            self.assertAlmostEqual(y0[1], y1[1])
            self.assertAlmostEqual(y0[2], y1[2])

    def test_RSP_from_cycle(self):
        # tests that cycles can be constructed from (branch_point, rotations)

        # in this example, each permutation is order 2 so any even combination
        # of monodromy paths should produce a cycle
        PF = RiemannSurfacePathFactory(self.X3)
        b = PF.branch_points
        cycle = [(b[0], 1), (b[1], -1)]
        gamma = PF.RiemannSurfacePath_from_cycle(cycle)
        x0 = gamma.get_x(0)
        y0 = gamma.get_y(0)
        x1 = gamma.get_x(1)
        y1 = gamma.get_y(1)
        self.assertAlmostEqual(x0, x1)
        self.assertAlmostEqual(y0[0], y1[0])
        self.assertAlmostEqual(y0[1], y1[1])

        cycle = [(b[0], 3), (b[1], 5)]
        gamma = PF.RiemannSurfacePath_from_cycle(cycle)
        x0 = gamma.get_x(0)
        y0 = gamma.get_y(0)
        x1 = gamma.get_x(1)
        y1 = gamma.get_y(1)
        self.assertAlmostEqual(x0, x1)
        self.assertAlmostEqual(y0[0], y1[0])
        self.assertAlmostEqual(y0[1], y1[1])

        # in this example, each permutation is order 3 and is the same
        # permutation (0 2 1) (in disjoint cycle notation)
        PF = RiemannSurfacePathFactory(self.X4)
        b = PF.branch_points
        cycle = [(b[0], 1), (b[1], 1), (b[2], 1)]
        gamma = PF.RiemannSurfacePath_from_cycle(cycle)
        x0 = gamma.get_x(0)
        y0 = gamma.get_y(0)
        x1 = gamma.get_x(1)
        y1 = gamma.get_y(1)
        self.assertAlmostEqual(x0, x1)
        self.assertAlmostEqual(y0[0], y1[0])
        self.assertAlmostEqual(y0[1], y1[1])
        self.assertAlmostEqual(y0[2], y1[2])

        cycle = [(b[0], 2), (b[1], 1)]
        gamma = PF.RiemannSurfacePath_from_cycle(cycle)
        x0 = gamma.get_x(0)
        y0 = gamma.get_y(0)
        x1 = gamma.get_x(1)
        y1 = gamma.get_y(1)
        self.assertAlmostEqual(x0, x1)
        self.assertAlmostEqual(y0[0], y1[0])
        self.assertAlmostEqual(y0[1], y1[1])
        self.assertAlmostEqual(y0[2], y1[2])

        cycle = [(b[0], 4), (b[1], -1), (b[2], 3)]
        gamma = PF.RiemannSurfacePath_from_cycle(cycle)
        x0 = gamma.get_x(0)
        y0 = gamma.get_y(0)
        x1 = gamma.get_x(1)
        y1 = gamma.get_y(1)
        self.assertAlmostEqual(x0, x1)
        self.assertAlmostEqual(y0[0], y1[0])
        self.assertAlmostEqual(y0[1], y1[1])
        self.assertAlmostEqual(y0[2], y1[2])

    def test_f2_from_cycle_issue(self):
        # there is an unlabeled issue with the construction of certain cycles
        # on the curve f2. it turned out that RiemannSurfacePath_from_cycle had
        # incorrectly created a ComplexPath from the segments
        #
        # this tests is here just to assert that no errors are raised or that
        # the calculation doesn't time out
        R = QQ['x,y']
        x,y = R.gens()
        f = -x**7 + 2*x**3*y + y**3

        X = RiemannSurface(f)
        PF = X.path_factory
        a_tuples = PF.skeleton.a_cycles()
        b_tuples = PF.skeleton.b_cycles()
        for cycle in a_tuples + b_tuples:
            gamma = PF.RiemannSurfacePath_from_cycle(cycle)

class TestGenus43(unittest.TestCase):
    def setUp(self):
        R = QQ['x,y']; x,y = R.gens()
        f43 = y**3*((y**4-16)**2 - 2*x) - x**12
        self.f43 = f43
        self.X43 = RiemannSurface(f43)
        _ = self.X43.discriminant_points

    def test_monodromy_path_construction(self):
        # is tests, these paths caused some problems
        X = self.X43
        b = X.discriminant_points
        gamma = X.path_factory.monodromy_path(b[2])
        gamma = X.path_factory.monodromy_path(b[5])

