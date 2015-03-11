from .test_abelfunctions import AbelfunctionsTestCase
from abelfunctions.integralbasis import (
    integral_basis,
)
import sympy
from sympy.abc import x,y

class TestIntegralBasis(AbelfunctionsTestCase):

    def test_f1(self):
        a = integral_basis(self.f1,x,y)
        b = [1, y*(x**2-x+1)/x**2]
        self.assertEqual(a,b)

    def test_f2(self):
        a = integral_basis(self.f2,x,y)
        b = [1, y/x, y**2/x**3]
        self.assertEqual(a,b)

    # def test_f3(self):
    #     a = integral_basis(self.f3,x,y)
    #     b = [1, y, (y**2-1)/(x-1), -y*(x - 4*y**2 + 3)/(4*x*(x - 1))]
    #     self.assertEqual(a,b)

    def test_f4(self):
        a = integral_basis(self.f4,x,y)
        b = [1, y/x]
        self.assertEqual(a,b)

    def test_f5(self):
        a = integral_basis(self.f5,x,y)
        b = [1, y, y**2, y**3, y*(y**3-1)/x, y**2*(y**3-1)/x**2]
        self.assertEqual(a,b)

    def test_f6(self):
        a = integral_basis(self.f6,x,y)
        b = [1, y, y**2/x, y**3/x]
        self.assertEqual(a,b)

    def test_f7(self):
        a = integral_basis(self.f7,x,y)
        b = [1, y, y**2]
        self.assertEqual(a,b)

    def test_f8(self):
        a = integral_basis(self.f8,x,y)
        b = [1, x*y, x*y**2, y**3*x, x**2*y**4, y**5*x**2]
        self.assertEqual(a,b)

    def test_f9(self):
        a = integral_basis(self.f9,x,y)
        b = [1, y, y**2]
        self.assertEqual(a,b)

    def test_f10(self):
        a = integral_basis(self.f10,x,y)
        b = [1, x*y, x**2*y**2, x**3*y**3]
        self.assertEqual(a,b)
