import unittest

from sympy import Poly, I
from sympy.abc import x,y,Z,T

from abelfunctions.integralbasis import (
    valuation, integral_basis)


# === test curves ===
# Example curves are from "Computing with Plane Algebraic Curves and Riemann 
# Surfaces" by Deconinck and Patterson, Lecture Notes in Mathematics, 2011
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


class TestIntegralBasis(unittest.TestCase):

    def setUp(self):
        pass


    def test_integral_basis(self):
        self.assertEqual(integral_basis(f1,x,y),
                         [1, y*(x**2 - x + 1)/x**2])
        self.assertEqual(integral_basis(f2,x,y),
                         [1, y/x, y**2/x**3])

        self.assertEqual(integral_basis(f4,x,y),
                         [1, y/x])
#         # long time
#         self.assertEqual(integral_basis(f5,x,y),
#                          [])
#         # long time
#        self.assertEqual(integral_basis(f6,x,y),
#                          [1, y, y**2/x, y**3/x])
        self.assertEqual(integral_basis(f7,x,y),
                         [1, y, y**2])
#        self.assertEqual(integral_basis(f8,x,y),
#                         [])
        self.assertEqual(integral_basis(f9,x,y),
                         [1, y, y**2])
        self.assertEqual(integral_basis(f10,x,y),
                         [1, x*y, x**2*y**2, x**3*y**3])

