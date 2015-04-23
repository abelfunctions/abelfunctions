import unittest
from abelfunctions.singularities import genus

import sympy
from sympy.abc import x,y

class TestGenus(unittest.TestCase):

    def setUp(self):
        pass

    def test_issue71(self):
        f = -x**5 + x + y**3
        g = genus(f,x,y)
        self.assertEqual(g,4)
