import abelfunctions
import sympy
import unittest

from sympy.abc import x,y

class AbelfunctionsTestCase(unittest.TestCase):
    def setUp(self):
        self.x = x
        self.y = y
        self.f1 = (x**2 - x + 1)*y**2 - 2*x**2*y + x**4
        self.f2 = -x**7 + 2*x**3*y + y**3
        self.f3 = (y**2-x**2)*(x-1)*(2*x-3) - 4*(x**2+y**2-2*x)**2
        self.f4 = y**2 + x**3 - x**2
        self.f5 = (x**2 + y**2)**3 + 3*x**2*y - y**3
        self.f6 = y**4 - y**2*x + x**2
        self.f7 = y**3 - (x**3 + y)**2 + 1
        self.f8 = x**2*y**6 + 2*x**3*y**5 - 1
        self.f9 = 2*x**7*y + 2*x**7 + y**3 + 3*y**2 + 3*y
        self.f10 = (x**3)*y**4 + 4*x**2*y**2 + 2*x**3*y - 1

        # various classifications
        self.f = [self.f1, self.f2, self.f3, self.f4, self.f5,
                  self.f6, self.f7, self.f8, self.f9, self.f10]
        self.monicf = [self.f2, self.f4, self.f5, self.f6, self.f7, self.f9]
        self.nonmonicf = [self.f1, self.f3, self.f8, self.f10]

class RiemannSurfaceTestCase(AbelfunctionsTestCase):
    def setUp(self):
        pass
