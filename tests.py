import sympy
import abelfunctions

from sympy.abc import x,y
from abelfunctions.riemann_surface import RiemannSurface

if __name__ == '__main__':
    f0 = y**3 - 2*x**3*y - x**8
    f1 = (x**2 - x + 1)*y**2 - 2*x**2*y + x**4
    f2 = -x**7 + 2*x**3*y + y**3
    f3 = (y**2-x**2)*(x-1)*(2*x-3) - 4*(x**2+y**2-2*x)**2
    f4 = y**2 + x**3 - x**2
    f5 = (x**2 + y**2)**3 + 3*x**2*y - y**3
    f6 = y**4 - y**2*x + x**2
    f7 = y**3 - (x**3 + y)**2 + 1
    f8 = (x**6)*y**3 + 2*x**3*y - 1
    f9 = 2*x**7*y + 2*x**7 + y**3 + 3*y**2 + 3*y
    f10 = (x**3)*y**4 + 4*x**2*y**2 + 2*x**3*y - 1
    f11 = y**2 - (x-2)*(x-1)*(x+1)*(x+2)
    f12 = x**4 + y**4 - 1

    f = f2

    X = RiemannSurface(f,x,y)
    print '=== base point ==='
    print X.base_point()

    print '\n=== base sheets ==='
    print X.base_sheets()

    PF = X.PF

    # print '\n=== plotting x-path ==='
    # t = linspace(0,1,32)
    # gamma.plot_x(t)
    print '\n=== branch_points ==='
    b = PF.branch_points()
    print b

    print '\n=== monpath0 ==='
    t = linspace(0,1,128)
    print '\n=== generating path ==='
    gamma = PF.monodromy_path(b[0])
    print '\n=== x-values ==='
    for ti in t:
        print ti, gamma.get_x(ti)

