import unittest
from abelfunctions.tests.test_abelfunctions import AbelfunctionsTestCase

from abelfunctions import RiemannSurface

class TestConstruction(AbelfunctionsTestCase):

    def test_places(self):
        X = RiemannSurface(self.f1)
        places = X(-3)
        for bi in X.branch_points:
            places = X(bi)

        X = RiemannSurface(self.f2)
        places = X(-3)
        for bi in X.branch_points:
            places = X(bi)
