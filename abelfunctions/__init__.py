"""
abelfunctions is a Python library for computing with Abelian functions,
algebraic curves, and solving integrable Partial Differential Equations.

The code is available as a git repository at

    https://github.com/cswiercz/abelfunctions

"""

try:
    import sage.all__sagemath_modules
except ImportError:
    pass

from abelfunctions.version import __version__
from abelfunctions.abelmap import AbelMap, Jacobian
from abelfunctions.homology import symmetrize_periods
from abelfunctions.puiseux import puiseux
from abelfunctions.riemann_constant_vector import RiemannConstantVector
from abelfunctions.riemann_surface import RiemannSurface
from abelfunctions.riemann_theta.riemann_theta import RiemannTheta

__all__ = [
    "__version__",
    "AbelMap",
    "Jacobian",
    "symmetrize_periods",
    "puiseux",
    "RiemannConstantVector",
    "RiemannSurface",
    "RiemannTheta",
]
