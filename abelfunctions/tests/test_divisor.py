from .test_abelfunctions import AbelfunctionsTestCase
from abelfunctions.divisor import (
    Place, DiscriminantPlace, RegularPlace, Divisor, ZeroDivisor
)


class TestDivisor(AbelfunctionsTestCase):

    def setUp(self):
        # create some dummy places
        self.P = RegularPlace(None,0,1,name='P')
        self.Q = RegularPlace(None,2,3,name='Q')
        self.R = RegularPlace(None,4,5,name='R')

    def test_addition(self):
        D1 = self.P + self.Q
        D2 = self.P + self.Q + self.R
        D3 = self.P + self.P
        d1 = D1.dict
        d2 = D2.dict
        d3 = D3.dict
        self.assertEqual(d1, {self.P:1, self.Q:1})
        self.assertEqual(d2, {self.P:1, self.Q:1, self.R:1})
        self.assertEqual(d3, {self.P:2})

    def test_subtraction(self):
        D1 = self.P + self.P - self.P
        P1 = D1.as_place()
        self.assertEqual(P1, self.P)

    def test_equality(self):
        D1 = self.P + self.Q
        D2 = self.Q + self.P
        self.assertEqual(D1,D2)

        D1 = self.P + self.P
        D2 = 2*self.P
        self.assertEqual(D1,D2)

        D1 = self.P - self.P
        D2 = ZeroDivisor(None)
        self.assertEqual(D1,D2)

    def test_associativity(self):
        D1 = (self.P + self.Q) + self.R
        D2 = self.P + (self.Q + self.R)
        self.assertEqual(D1,D2)

    def test_zero(self):
        D = Divisor(None,0)
        self.assertTrue(D.is_zero())

        D = self.P - self.P
        self.assertTrue(D.is_zero())

    def test_multiplicity(self):
        D1 = 2*self.P
        D2 = 0*self.P
        d1 = D1.dict
        self.assertEqual(d1, {self.P:2})
        self.assertTrue(D2.is_zero())
