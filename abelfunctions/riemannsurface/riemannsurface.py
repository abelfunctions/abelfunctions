"""
Riemann Surfaces
"""
import numpy
import scipy
import scipy.linalg as la
import scipy.integrate
import sympy


from abelfunctions.monodromy import monodromy, show_paths
from abelfunctions.homology import homology
from abelfunctions.riemannsurface.riemannsurface_path import (
    path_segments_from_cycle,
    RiemannSurfacePath,
    )
from abelfunctions.riemannsurface.riemannsurface_point import (
    RiemannSurfacePoint,
    )
from abelfunctions.singularities import genus
from abelfunctions.differentials import differentials
from abelfunctions.utilities import cached_function

class RiemannSurface(object):
    """
    Class for defining a Riemann surface corresponding to a plane algebraic
    curve.
    """
    def __init__(self,f,x,y,base_point=None,base_sheets=None):
        self.f = f
        self.x = x
        self.y = y
        self._base_point=base_point
        self._base_sheets=base_sheets

    def __repr__(self):
        return "Riemann surface defined by the algebraic curve %s." %(self.f)

    def __call__(self, x, y):
        return RiemannSurfacePoint(self, x, y)

    def point(self,x0,y0):
        """
        Returns the point on the Riemann surface closest to `(x0,y0)`. (In the
        event that floating point precision is used to calculate the
        coordinates `(x0,y0)`.
        """
        return RiemannSurfacePoint(self,x0,y0)

    def monodromy(self):
        """
        """
        return monodromy(self.f,self.x,self.y,base_point=self._base_point,
                         base_sheets=self._base_sheets)

    def monodromy_graph(self):
        """
        """
        return self.monodromy()[4]

    def base_point(self):
        return self.monodromy()[0]

    def base_sheets(self):
        return self.monodromy()[1]

    def base_lift(self):
        return self.base_sheets()

    def branch_points(self):
        return self.monodromy()[2]

    def monodromy_group(self):
        return self.monodromy()[3]

    def homology(self,verbose=False):
        # always compute the verbose version of homology
        d = homology(self.f,self.x,self.y,
                     base_point=self._base_point,
                     base_sheets=self._base_sheets,
                     verbose=True)

        if verbose:
            return d
        else:
            return (d['a-cycles'],d['b-cycles'])

    def _c_cycles(self):
        return self.homology(verbose=True)['c-cycles']

    def _linear_combinations(self):
        return self.homology(verbose=True)['linearcombinations']

    def holomorphic_differentials(self):
        """
        Returns the basis of holomorphic differentials defined on the
        Riemann surface.
        """
        return differentials(self.f, self.x, self.y)

    def differentials(self):
        """
        Alias for RiemannSurface.holomorphic_differentials().
        """
        return self.holomorphic_differentials()

    def genus(self):
        """
        Return the genus of the Riemann surface.
        """
        return genus(self.f, self.x, self.y)

    @cached_function
    def integrate(self, omega, x, y, path, **kwds):
        """
        Integrates the differential `omega`, defined on the
        Riemann surface, on the RiemannSurfacePath `path`.

        Input:

        - `omega,x,y`: a Sympy funciton in the variables `x` and `y`

        - `path`: a RiemannSurfacePath defined on the Riemann surface
        """
        return path.integrate(omega,x,y)

    def period_matrix(self):
        """Returns the period matrix corresponding to the Riemann Surface.
        """
        differentials = self.holomorphic_differentials()
        base_point = self.base_point()
        base_sheets = self.base_sheets()
        G = self.monodromy_graph()
        g = self.genus()
        x = self.x
        y = self.y

        # store the values of the integrals of the c-cycles but only for
        # the ones where the linear combination index is non-zero. that
        # is, only compute the integrals of the c-cycles that are
        # actually used
        c_cycles = self._c_cycles()
        m = len(c_cycles)
        lincombs = self._linear_combinations()
        c_integrals = dict.fromkeys(
            range(m), numpy.zeros(len(differentials), dtype=numpy.complex)
            )
        c_needed = [j for i in range(2*g) for j in range(m)
                    if lincombs[i,j] != 0]

        for k in c_needed:
            c_cycle = c_cycles[k]
            gamma = RiemannSurfacePath(self,(base_point,base_sheets),
                                       cycle=c_cycle)
            c_integrals[k] = [
                self.integrate(omega, x, y, gamma) for omega in differentials
                ]

        # now take appropriate linear combinations to compute the
        # integrals of the differentials around the a- and b-cycles
        tau = numpy.zeros((g,2*g),dtype=numpy.complex)
        for i in range(g):
            for j in range(2*g):
                tau[i][j] = sum(
                    lincombs[j,k] * c_integrals[k][i] for k in range(m)
                    )

        A = tau[:g,:g]
        B = tau[:g,g:]
        return A,B

    def show_paths(self):
        """
        """
        G = self.monodromy_graph()
        show_paths(G)



if __name__ == '__main__':
    from sympy.abc import x,y
    from abelfunctions.riemannsurface.riemannsurface_path import polyroots

    f0 = y**3 - 2*x**3*y - x**8  # Klein curve

    f1 = (x**2 - x + 1)*y**2 - 2*x**2*y + x**4
    f2 = -x**7 + 2*x**3*y + y**3
    f3 = (y**2-x**2)*(x-1)*(2*x-3) - 4*(x**2+y**2-2*x)**2
    f4 = y**2 + x**3 - x**2
    f5 = (x**2 + y**2)**3 + 3*x**2*y - y**3
    f6 = y**4 - y**2*x + x**2   # case with only one finite disc pt
    f7 = y**3 - (x**3 + y)**2 + 1
    f8 = (x**6)*y**3 + 2*x**3*y - 1
    f9 = 2*x**7*y + 2*x**7 + y**3 + 3*y**2 + 3*y
    f10= (x**3)*y**4 + 4*x**2*y**2 + 2*x**3*y - 1  # genus 3

    f11 = y**2 - (x-2)*(x-1)*(x+1)*(x+2)  # simple genus one hyperelliptic
    f12 = x**4 + y**4 - 1


    f = f2

#     # f2
#     f = f2
#     I = 1.0j
#     base_point = -1.44838920232100
#     base_sheets = [-3.20203812255,
#                     1.60101906127-1.26997391750*I,
#                     1.60101906127+1.26997391750*I]

#     # f10
#     f = f10
#     I = 1.0j
#     base_point = -1.43572291547089
#     base_sheets = [
#         -1.93155860973,
#          -0.141326328588,
#          1.03644246916-.404482364824*I,
#          1.03644246916+.404482364824*I
#          ]

    X = RiemannSurface(f,x,y)#,base_point=base_point,base_sheets=base_sheets)

    print "\n\tRS"
    print X

    print "\n\tRS: monodromy"
    base_point, base_sheets, branch_points, mon, G = X.monodromy()
    print "\nbase_point:"
    print base_point
    print "\nbase_sheets:"
    for s in base_sheets: print s
    print "\nbranch points:"
    for b in branch_points: print b
    print "\nmonodromy group:"
    for m in mon:
        print m

    print "\n\tRS: homology"
    hom = X.homology(verbose=True)
    for key,value in hom.items():
        print key
        print value
        print

    print "\n\tRS: computing differentials"
    diffs = X.holomorphic_differentials()

    print "\n\tRS: period matrix"
    A,B = X.period_matrix()
    Omega = numpy.dot(la.inv(A),B)
    print "\n\tA = "
    print A
    print "\n\tB = "
    print B
    print "\n\tOmega (abelfunctions)"
    print Omega
    print
