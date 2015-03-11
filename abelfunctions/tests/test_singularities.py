from abelfunctions.singularities import singularities
from sympy import sympify, RootOf

from .test_abelfunctions import AbelfunctionsTestCase

class TestSingularities(AbelfunctionsTestCase):

    def test_f1(self):
        s = singularities(self.f1,self.x,self.y)
        s = sympify(s)
        s_actual = sympify(
            [((0,0,1),(2,2,1)),
             ((0,1,0),(2,1,2))]
        )
        self.assertItemsEqual(s,s_actual)

    def test_f2(self):
        s = singularities(self.f2,self.x,self.y)
        s = sympify(s)
        s_actual = sympify(
            [((0,0,1),(3,4,2)),
             ((0,1,0),(4,9,1))]
        )
        self.assertItemsEqual(s,s_actual)

    def test_f3(self):
        s = singularities(self.f3,self.x,self.y)
        s = sympify(s)
        s_actual = sympify(
            [((0,0,1),(2,1,2)),
             ((1,-1,1),(2,1,2)),
             ((1,1,1),(2,1,2))]
        )
        self.assertItemsEqual(s,s_actual)

    def test_f4(self):
        s = singularities(self.f4,self.x,self.y)
        s = sympify(s)
        s_actual = sympify(
            [((0,0,1),(2,1,2))]
        )
        self.assertItemsEqual(s,s_actual)

    def test_f5(self):
        pass

    def test_f6(self):
        s = singularities(self.f6,self.x,self.y)
        s = sympify(s)
        s_actual = sympify(
            [((0,0,1),(2,2,2)),
             ((1,0,0),(2,2,2))]
        )
        self.assertItemsEqual(s,s_actual)

    def test_f7(self):
        s = singularities(self.f7,self.x,self.y)
        s = sympify(s)
        s_actual = sympify(
            [((0,1,0),(3,6,3))]
        )
        self.assertItemsEqual(s,s_actual)

    def test_f8(self):
        s = singularities(self.f8,self.x,self.y)
        s = sympify(s)
        s_actual = sympify(
            [((0,1,0),(2,4,2)),
             ((1,0,0),(5,14,1))]
        )
        self.assertItemsEqual(s,s_actual)

    def test_f9(self):
        s = singularities(self.f9,self.x,self.y)
        s = sympify(s)
        s_actual = sympify(
            [((0,1,0),(5,12,1))]
        )
        self.assertItemsEqual(s,s_actual)

    def test_f10(self):
        s = singularities(self.f10,self.x,self.y)
        s = sympify(s)
        s_actual = sympify(
            [((0,1,0),(3,6,1)),
             ((1,0,0),(4,6,4))]
        )
        self.assertItemsEqual(s,s_actual)
