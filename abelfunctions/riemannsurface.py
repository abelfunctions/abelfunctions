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
from abelfunctions.riemannsurface_path import (
    path_around_branch_point,
    path_around_infinity,
    RiemannSurfacePath,
    )
from abelfunctions.riemannsurface_point import RiemannSurfacePoint
from abelfunctions.singularities import genus
from abelfunctions.differentials import differentials
from abelfunctions.utilities import cached_function


import pdb

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


    def homology(self):
        return homology(self.f,self.x,self.y)


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


    def cycles(self):
        """
        """
        return self.homology()

    @cached_function
    def cycle_paths(self):
        """
        Returns a path in the complex x-plane of the x-points in the
        monodromy path.

        Input:

        - i: the index of the c-cycle

        Output:

        - a RiemannSurfacePath parameterizing the c-cycle
        """
        abs = numpy.abs

        # get the cycle data from homology: each c-cycle is a list of
        # alternating sheet numbers s_k and branch point / number of
        # rotations tuples (b_{i_k}, n_k)
        a_cycles, b_cycles = self.homology()
        cycles = a_cycles + b_cycles

        G = self.monodromy_graph()
        root = G.node[0]['root']

        cycle_paths = []
        for cycle in cycles:
            path_segment_data = []

            # For each (branch point, rotation number) pair appearing in
            # the cycle compute the path segments the complex x-plane
            # going around the given branch points a number of times equal
            # to the rotation number. Add these segments to the list of
            # path segemnts.
            for (bpt, rot) in cycle[1::2]:
                if bpt == sympy.oo:
                    bpt_path_segdat = path_around_infinity(G,rot)
                else:
                    bpt_index = [key for key,data in G.nodes(data=True)
                                 if abs(data['value']-bpt) < 1e-15][0]
                    bpt_path_segdat = path_around_branch_point(G,bpt_index,rot)
                path_segment_data.extend(bpt_path_segdat)

            # if a custom base point is provided then add the
            # appropriate path segment data
            if self._base_point:
                seg = (self.base_point(), G.node[0]['basepoint'])
                path_segment_data = [seg] + path_segment_data + \
                    [tuple(reversed(seg))]

            # Construct the RiemannSurfacePath and append to path list
            x0 = self.base_point()
            y0 = self.base_lift()
            gamma = RiemannSurfacePath(self, (x0,y0),
                                       path_segment_data=path_segment_data)
            cycle_paths.append(gamma)

        return cycle_paths


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
        """
        Returns the period matrix `\tau = (A \; B)` where `A_{ij}` is
        the integral of the `j`th holomorphic differential basis element
        about the cycle `a_i`. (`B` is defined in the same way about the
        `b`-cycles.)

        Note: this function computes the integrals of the
        differentials over each of the c-cycles and then uses the
        homology data to determine which linear combination of
        c-cycles integrals gives the a- and b-cycles integrals.
        """
        cycles = self.cycle_paths()

        differentials = self.holomorphic_differentials()
        g = self.genus()
        x = self.x
        y = self.y

        A = []
        B = []
        for i in xrange(g):
            omega = differentials[i]
            Ai = []
            Bi = []
            for j in xrange(g):
                print "A[%d,%d]"%(i,j)
                integral = self.integrate(omega,x,y,cycles[j])
                Ai.append(integral)

                print "B[%d,%d]"%(i,j)
                integral = self.integrate(omega,x,y,cycles[j+g])
                Bi.append(integral)

            A.append(Ai)
            B.append(Bi)

        # need to transpose...?
        A = numpy.matrix(A,dtype=numpy.complex)
        B = numpy.matrix(B,dtype=numpy.complex)
        return (A,B)


    def show_paths(self):
        """
        """
        G = self.monodromy_graph()
        show_paths(G)



if __name__ == '__main__':
    from sympy.abc import x,y
    from abelfunctions.riemannsurface_path import polyroots

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

    # f2
    I = 1.0j
    base_point = -1.44838920232100
    base_sheets = [-3.20203812255,
                    1.60101906127-1.26997391750*I,
                    1.60101906127+1.26997391750*I]

    X = RiemannSurface(f,x,y,base_point=base_point,base_sheets=base_sheets)

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
    for m in mon: print m

    print "\n\tRS: homology"
    a_cycles, b_cycles = X.homology()
    for a in a_cycles:
        print a
    print
    for b in b_cycles:
        print b

    print "\n\tRS: computing cycles"
    cycles = X.cycles()

    print "\n\tRS: computing paths"
    paths = X.cycle_paths()

    print "\n\tRS: computing differentials"
    diffs = X.holomorphic_differentials()

#     print "\n\tRS: computing paths"
#     paths = X.cycle_paths()

#     print "\n\tRS: period matrix"
#     A,B = X.period_matrix()
#     Omega = numpy.dot(la.inv(A),B)
#     print "\n\tA = "
#     print A
#     print "\n\tB = "
#     print B
#     print "\n\tOmega (abelfunctions)"
#     print Omega
#     print

    paths[3].plot_differential(diffs[1],x,y,N=1024)
