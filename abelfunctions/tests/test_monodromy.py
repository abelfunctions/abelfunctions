import unittest
import sympy

from sympy import sympify
from sympy.abc import x,y

from abelfunctions.monodromy import (
    monodromy_graph,
    monodromy,
    Permutation,
    )

# === test curves ===
#
# Example curves are from "Computing with Plane Algebraic Curves and
# Riemann Surfaces" by Deconinck and Patterson, Lecture Notes in
# Mathematics, 2011
f1 = (x**2 - x + 1)*y**2 - 2*x**2*y + x**4                   # p. 71
f2 = y**3 + 2*x**3*y - x**7                                  # p. 73
f3 = (y**2-x**2)*(x-1)*(2*x-3) - 4*(x**2+y**2-2*x)**2        # p. 75
f4 = y**2 + x**3 - x**2                                      # p. 82
f5 = (x**2 + y**2)**3 + 3*x**2*y - y**3                      # p. 84
f6 = y**4 - y**2*x + x**2                                    # p. 85
f7 = y**3 - (x**3 + y)**2 + 1                                # p. 85

# example singular curves:
f8 = x**2*y**6 + 2*x**3*y**5 - 1
f9 = 2*x**7*y + 2*x**7 + y**3 + 3*y**2 + 3*y
f10= (x**3)*y**4 + 4*x**2*y**2 + 2*x**3*y - 1

class TestSingularities(unittest.TestCase):

    def setUp(self):
        pass

    def test_monodromy_maple(self):
        """
        The monodromy is compared against the output of Maple. Maple's
        default choice of base point and ordering of base sheets is
        given to monodromy().
        """
        I = 1.0j

        # f3
        base_point = -0.140000000000
        base_sheets = [-.498276346452241 -.283688768607067*I,
                         -.498276346452241+.283688768607067*I,
                         .498276346452241-.283688768607067*I,
                         .498276346452241+.283688768607067*I]
        base_point, base_sheets, branch_points, mon, G = \
            monodromy(f3,x,y,base_point=base_point,base_sheets=base_sheets)
        branch_points_actual = [(1.75-0.3227486121839514j),
                                (1.5+0j),
                                (-0.1+0j),
                                (1.75+0.3227486121839514j)]
        mon_actual = map(Permutation,[[0, 2, 1, 3],
                                      [2, 3, 0, 1],
                                      [1, 0, 3, 2],
                                      [3, 1, 2, 0]])
        self.assertEqual(branch_points,branch_points_actual)
        self.assertEqual(mon,mon_actual)

if __name__ == '__main__':
    unittest.main()
