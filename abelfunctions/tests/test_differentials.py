import unittest
import sympy 

from sympy import sympify
from sympy.abc import x,y

from abelfunctions.differentials import (
    differentials,)

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

# example non-singular curves
f11= x**4 + y**4 - 1

class TestDifferentials(unittest.TestCase):

    def setUp(self):
        pass

    def test_differentials(self):
        self.assertItemsEqual(
            differentials(f1,x,y),
            [],
            )

        self.assertItemsEqual(
            differentials(f2,x,y),
            [x**3/(2*x**3 + 3*y**2), x*y/(2*x**3 + 3*y**2)],
            )

        self.assertItemsEqual(
            differentials(f3,x,y),
            [],
            )

        self.assertItemsEqual(
            differentials(f4,x,y),
            [],
            )

        # counter += 1
        # self.assertItemsEqual(
        #     differentials(f5,x,y),
        #     [(x**2 + y**2)/(3*x**2 - 3*y**2 + 6*y*(x**2 + y**2)**2)],
        #     )

        # self.assertItemsEqual(
        #     differentials(f6,x,y),
        #     [],
        #     )

        # # f7 takes too long
        # self.assertItemsEqual(
        #     differentials(f8,x,y),
        #     [],
        #     )

        # self.assertItemsEqual(
        #     differentials(f9,x,y),
        #     [x**5/(2*x**7 + 3*y**2 + 6*y + 3),
        #      x**4/(2*x**7 + 3*y**2 + 6*y + 3),
        #      x**3/(2*x**7 + 3*y**2 + 6*y + 3),
        #      x**2*y/(2*x**7 + 3*y**2 + 6*y + 3),
        #      x**2/(2*x**7 + 3*y**2 + 6*y + 3),
        #      x*y/(2*x**7 + 3*y**2 + 6*y + 3),
        #      x/(2*x**7 + 3*y**2 + 6*y + 3),
        #      y/(2*x**7 + 3*y**2 + 6*y + 3),
        #      1/(2*x**7 + 3*y**2 + 6*y + 3)],
        #     )

        # self.assertItemsEqual(
        #     differentials(f10,x,y),
        #     [x*y/(4*x**3*y**3 + 2*x**3 + 8*x**2*y),
        #      x/(4*x**3*y**3 + 2*x**3 + 8*x**2*y),
        #      1/(4*x**3*y**3 + 2*x**3 + 8*x**2*y)],
        #     )

        # self.assertItemsEqual(
        #     differentials(f11,x,y),
        #     [x/(4*y**3), 1/(4*y**2), 1/(4*y**3)],
        #     )


          
