"""
abelfunctions is a Python library for computing with Abelian functions, 
algebraic curves, and solving integrable Partial Differential Equations.

The code is available as a git repository at

    https://github.com/cswiercz/abelfunctions

"""
__version__ = "beta"

from puiseux import puiseux
from integralbasis import integral_basis
from singularities import singularities, homogenize, _transform, genus
from differentials import differentials
from monodromy import monodromy #, show_paths
from homology import homology
from riemannsurface_point import RiemannSurfacePoint
from riemannsurface_path import RiemannSurfacePath #, path_around_branch_point
from riemannsurface import RiemannSurface
from riemanntheta import RiemannTheta



