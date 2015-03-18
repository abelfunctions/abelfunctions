from .test_abelfunctions import AbelfunctionsTestCase
from abelfunctions.differentials import Differential, differentials

import sympy
from sympy.abc import x,y

# class TestDifferentials(AbelfunctionsTestCase):

#     def test_f1(self):
#         a = differentials(self.f1,x,y)
#         a = map(lambda omega: omega.as_sympy_expr(), a)
#         b = []
#         self.assertEqual(a,b)

#     def test_f2(self):
#         a = differentials(self.f2,x,y)
#         a = map(lambda omega: omega.as_sympy_expr(), a)
#         b = [x*y/(2*x**3+3*y**2), x**3/(2*x**3+3*y**2)]
#         self.assertEqual(a,b)

#     def test_f4(self):
#         a = differentials(self.f4,x,y)
#         a = map(lambda omega: omega.as_sympy_expr(), a)
#         b = []
#         self.assertEqual(a,b)

#     def test_f5(self):
#         a = differentials(self.f5,x,y)
#         a = map(lambda omega: omega.as_sympy_expr(), a)
#         b = [(x**2+y**2)/(2*x**4*y+4*x**2*y**3+2*y**5+x**2-y**2)]
#         self.assertEqual(a,b)

#     def test_f7(self):
#         a = differentials(self.f7,x,y)
#         a = map(lambda omega: omega.as_sympy_expr(), a)
#         b = [1/(2*x**3-3*y**2+2*y),
#              y/(2*x**3-3*y**2+2*y),
#              x/(2*x**3-3*y**2+2*y),
#              x**2/(2*x**3-3*y**2+2*y)]
#         self.assertEqual(a,b)

#     def test_f8(self):
#         a = differentials(self.f8,x,y)
#         a = map(lambda omega: omega.as_sympy_expr(), a)
#         b = []
#         self.assertEqual(a,b)

