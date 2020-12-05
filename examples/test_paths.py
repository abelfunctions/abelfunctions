print "=== Riemann surface path ==="
import sympy
from sympy.abc import x,y
from abelfunctions.monodromy import (
    monodromy_graph,
    show_paths,
    monodromy
    )
from abelfunctions.riemannsurface_path import (
    RiemannSurfacePath,
    path_around_branch_point,
    path_around_infinity,
    )

f1 = (x**2 - x + 1)*y**2 - 2*x**2*y + x**4
f2 = -x**7 + 2*x**3*y + y**3
f3 = (y**2-x**2)*(x-1)*(2*x-3) - 4*(x**2+y**2-2*x)**2
f4 = y**2 + x**3 - x**2
f5 = (x**2 + y**2)**3 + 3*x**2*y - y**3
f6 = y**4 - y**2*x + x**2   # case with only one finite disc pt
f7 = y**3 - (x**3 + y)**2 + 1
f8 = (x**6)*y**3 + 2*x**3*y - 1
f9 = 2*x**7*y + 2*x**7 + y**3 + 3*y**2 + 3*y
f10= (x**3)*y**4 + 4*x**2*y**2 + 2*x**3*y - 1

f = f2

G = monodromy_graph(f,x,y)
show_paths(G)
path_segments = path_around_branch_point(G,2,1)

base_point = G.nodes[0]['basepoint']
base_sheets = G.nodes[0]['baselift']

x0,y0 = base_point, base_sheets
gamma = RiemannSurfacePath((f,x,y),(x0,y0),path_segments=path_segments)
gamma.plot(Npts=128)

base_point, base_sheets, branch_points, mon, G = monodromy(f,x,y)

