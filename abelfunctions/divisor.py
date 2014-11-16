r""" Places and Divisors :mod:`divisor`
==================================

A module defining places and divisors on a Riemann surface. A *regular
affine place* is simply given as an :math:`(x_0,y_0)` tuple on the curve
:math:`C : f(x,y) = 0` from which the Riemann surface is defined.

Classes
-------

.. autosummary::

    Divisor
    Place

Functions
---------

Examples
--------

Contents
--------

"""

class Place(object):
    def __init__(self, RS, x, y):
        self.RS = RS
        self.x = x
        self.y = y

    def __repr__(self):
        return str((self.x,self.y))

    def __getitem__(self, idx):
        """Let's you do Place[idx].

        Place[0] - xproj == Place.x
        Place[1] - yproj == Place.y
        """
        if idx == 0:
            return self.x
        elif idx == 1:
            return self.y
        else:
            raise IndexError()

    def is_discriminant(self):
        return False

class DiscriminantPlace(Place):
    def __init__(self, RS, P):
        self.puiseux_series = P
        self.t = P.t
        x = P.x0
        y = P.eval_y(0)
        Place.__init__(self,RS,x,y)

    def __repr__(self):
        return str(self.puiseux_series)

    def is_discriminant(self):
        return True


class Divisor(object):
    pass
