from abelfunctions import RiemannSurface
from abelfunctions.abelmap import Jacobian

from sage.rings.rational_field import Q as QQ


def test_bug():
    R = QQ["x,y"]
    x, y = R.gens()

    f11 = x**2 * y**3 - x**4 + 1
    X11 = RiemannSurface(f11)
    X11.riemann_matrix()
    X11.genus()
    Jacobian(X11)
