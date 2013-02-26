import unittest
import sympy 

from sympy.abc import x,y

from abelfunctions.singularities import (
    singularities,)

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


class TestDifferentials(unittest.TestCase):

    def setUp(self):
        pass

    def test_singularities(self):
        S1 = singularities(f1,x,y)
        S2 = singularities(f2,x,y)
        S3 = singularities(f3,x,y)
        S4 = singularities(f4,x,y)
        S5 = singularities(f5,x,y)
        S6 = singularities(f6,x,y)
        S7 = singularities(f7,x,y)
        S8 = singularities(f8,x,y)
        S9 = singularities(f9,x,y)
        S10= singularities(f10,x,y)

        rt5 = [rt for rt,_ in sympy.roots(x**2 + 1, x).iteritems()]

        S1act = [
            ((0,0,1),(2,2,1)),
            ((0,1,0),(2,1,2))
            ]
        S2act = [
            ((0,0,1),(3,4,2)),
            ((0,1,0),(4,9,1))
            ]
        S3act = [
            ((0,0,1),(2,1,2)),
            ((1,-1,1),(2,1,2)),
            ((1,1,1),(2,1,2))
            ]
        S4act = [
            ((0,0,1),(2,1,2))
            ]
        S5act = [
            ((0,0,1),(3,3,3)),
            ((rt5[0],1,0),(3,3,3)),
            ((rt5[1],1,0),(3,3,3))
            ]
        S6act = [
            ((0,0,1),(2,2,2)),
            ((1,0,0),(2,2,2))
            ]
        S7act = [((0,1,0),(3,6,3))]
        S8act = [
            ((0,1,0),(6,21,3)),
            ((1,0,0),(3,7,2))
            ]
        S9act = [((0,1,0),(5,12,1))]
        S10act= [
            ((0,1,0),(3,6,1)),
            ((1,0,0),(4,6,4))
            ]

        self.assertItemsEqual(S1,S1act)
        self.assertItemsEqual(S2,S2act)
        self.assertItemsEqual(S3,S3act)
        self.assertItemsEqual(S4,S4act)
        self.assertItemsEqual(S5,S5act)
        self.assertItemsEqual(S6,S6act)
        self.assertItemsEqual(S7,S7act)
        self.assertItemsEqual(S8,S8act)
        self.assertItemsEqual(S9,S9act)
        self.assertItemsEqual(S10,S10act)
        

if __name__ == '__main__':
    unittest.main()
