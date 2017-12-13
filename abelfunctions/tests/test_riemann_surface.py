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

    def test_differential_property(self):
        X = RiemannSurface(self.f2)

        # Test initial value
        self.assertIsNone(X._user_differentials)

        # Test default property
        diffs = X.differentials
        self.assertTrue(len(diffs) == X.genus())

        # Test invalid set
        with self.assertRaises(ValueError):
            X.differentials = []

        # Test valid set
        X.differentials = diffs[::-1]
        self.assertIsNotNone(X._user_differentials)
        for a, b in zip(diffs[::-1], X.differentials):
            self.assertIs(a, b)

        # Test clear
        X.differentials = None
        self.assertIsNone(X._user_differentials)