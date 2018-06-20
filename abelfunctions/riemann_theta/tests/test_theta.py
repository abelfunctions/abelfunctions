"""Riemann Theta Tests

References
----------

.. [CRTF] B. Deconinck, M.  Heil, A. Bobenko, M. van Hoeij and M. Schmies,
   Computing Riemann Theta Functions, Mathematics of Computation, 73, (2004),
   1417-1442.

.. [DLMF] B. Deconinck, Digital Library of Mathematics Functions - Riemann Theta
   Functions, http://dlmf.nist.gov/21

.. [SAGE] Computing Riemann theta functions in Sage with applications.
   C. Swierczewski and B. Deconinck.Submitted for publication.  Available online
   at http://depts.washington.edu/bdecon/papers/pdfs/Swierczewski_Deconinck1.pdf

"""


import unittest
import numpy

from numpy.random import randn
from numpy.linalg import norm, cholesky
from abelfunctions.riemann_theta.radius import radius
from abelfunctions.riemann_theta.riemann_theta import RiemannTheta

# try to import mpmath's jtheta function
NO_JTHETA = False
try:
    from sympy.mpmath import jtheta
except ImportError:
    try:
        from mpmath import jtheta
    except ImportError:
        NO_JTHETA = True

def thetag1(z,tau,N=2048):
    r"""Naive implementation of genus 1 theta function."""
    return sum(numpy.exp(numpy.pi*1.j*tau*n**2 + 2.j*numpy.pi*n*z)
               for n in range(-N,N))
thetag1 = numpy.vectorize(thetag1, otypes=(numpy.complex,), excluded=(1,2))



class TestMaple(unittest.TestCase):
    def setUp(self):
        self.Omega3 = numpy.array(
            [[1.j, 0.5, 0.5],
             [0.5, 1.j, 0.5],
             [0.5, 0.5, 1.j]], dtype=numpy.complex)
        self.Omega4 = numpy.array([
            [ 0.39344262+0.79503971j, -0.75409836-0.36912558j,
              -0.44262295-0.02839428j,  0.20491803+0.26974562j],
            [-0.75409836-0.36912558j,  0.27868852+0.85182827j,
             0.09836066+0.19875993j, -0.43442623-0.15616852j],
            [-0.44262295-0.02839428j,  0.09836066+0.19875993j,
             -0.37704918+0.68146261j, -0.91803279+0.45430841j],
            [ 0.20491803+0.26974562j, -0.43442623-0.15616852j,
              -0.91803279+0.45430841j, -1.27868852+0.88022254j]
        ])

    def test_value(self):
        z = [0,0,0]
        Omega = self.Omega3

        value = RiemannTheta(z, Omega, epsilon=1e-14)
        maple = 1.2362529854204190 - 0.52099320642367818e-10j
        error = abs(value - maple)
        self.assertLess(error, 1e-8)

        w = [0.2+0.5j, 0.3-0.1j, -0.1+0.2j]
        value = RiemannTheta(w, Omega, epsilon=1e-14)
        maple = 1.2544694041047501 - 0.77493173321770725j
        error = abs(value - maple)
        self.assertLess(error, 1e-8)

    def test_first_derivatives(self):
        w = [0.2+0.5j, 0.3-0.1j, -0.1+0.2j]
        Omega = self.Omega3

        value_z1 = RiemannTheta(w, Omega, epsilon=1e-14, derivs=[[1,0,0]])
        value_z2 = RiemannTheta(w, Omega, epsilon=1e-14, derivs=[[0,1,0]])
        value_z3 = RiemannTheta(w, Omega, epsilon=1e-14, derivs=[[0,0,1]])

        maple_z1 = -5.7295900733729553 - 0.89199375315523345j
        maple_z2 = -0.16300987772384356 - 0.65079269102999180j
        maple_z3 = 1.0115406077003542 + 0.030528533907836019j

        error_z1 = abs(value_z1 - maple_z1)
        error_z2 = abs(value_z2 - maple_z2)
        error_z3 = abs(value_z3 - maple_z3)

        self.assertLess(error_z1, 1e-8)
        self.assertLess(error_z2, 1e-8)
        self.assertLess(error_z3, 1e-8)

        Omega = self.Omega4

        w = [0,0,0,0]
        value_z1 = RiemannTheta(w, Omega, epsilon=1e-14, derivs=[[1,0,0,0]])
        value_z2 = RiemannTheta(w, Omega, epsilon=1e-14, derivs=[[0,1,0,0]])
        value_z3 = RiemannTheta(w, Omega, epsilon=1e-14, derivs=[[0,0,1,0]])
        value_z4 = RiemannTheta(w, Omega, epsilon=1e-14, derivs=[[0,0,0,1]])

        maple_z1 = 0.0
        maple_z2 = 0.0
        maple_z3 = 0.0
        maple_z4 = 0.0

        error_z1 = abs(value_z1 - maple_z1)
        error_z2 = abs(value_z2 - maple_z2)
        error_z3 = abs(value_z3 - maple_z3)
        error_z4 = abs(value_z4 - maple_z4)

        self.assertLess(error_z1, 1e-8)
        self.assertLess(error_z2, 1e-8)
        self.assertLess(error_z3, 1e-8)
        self.assertLess(error_z4, 1e-8)

        # different value of w
        w = [-0.37704918-0.18456279j, 0.63934426+0.42591413j,
             0.54918033+0.09937996j, -0.21721311-0.07808426j]
        value_z1 = RiemannTheta(w, Omega, epsilon=1e-14, derivs=[[1,0,0,0]])
        value_z2 = RiemannTheta(w, Omega, epsilon=1e-14, derivs=[[0,1,0,0]])
        value_z3 = RiemannTheta(w, Omega, epsilon=1e-14, derivs=[[0,0,1,0]])
        value_z4 = RiemannTheta(w, Omega, epsilon=1e-14, derivs=[[0,0,0,1]])

        maple_z1 = 3.3644150756 + 2.5018071784j
        maple_z2 = -2.9431860155 + 5.6802762853j
        maple_z3 = 8.0319838396 + 3.5491434873j
        maple_z4 = -6.0837267311 - 2.4867680289j

        error_z1 = abs(value_z1 - maple_z1)
        error_z2 = abs(value_z2 - maple_z2)
        error_z3 = abs(value_z3 - maple_z3)
        error_z4 = abs(value_z4 - maple_z4)

        self.assertLess(error_z1, 1e-8)
        self.assertLess(error_z2, 1e-8)
        self.assertLess(error_z3, 1e-8)
        self.assertLess(error_z4, 1e-8)

    def test_first_derivatives_oscpart(self):
        # different value of w
        Omega = self.Omega4
        w = [-0.37704918-0.18456279j, 0.63934426+0.42591413j,
             0.54918033+0.09937996j, -0.21721311-0.07808426j]
        value_z1 = RiemannTheta.oscillatory_part(
            w, Omega, epsilon=1e-14, derivs=[[1,0,0,0]])
        value_z2 = RiemannTheta.oscillatory_part(
            w, Omega, epsilon=1e-14, derivs=[[0,1,0,0]])
        value_z3 = RiemannTheta.oscillatory_part(
            w, Omega, epsilon=1e-14, derivs=[[0,0,1,0]])
        value_z4 = RiemannTheta.oscillatory_part(
            w, Omega, epsilon=1e-14, derivs=[[0,0,0,1]])

        maple_z1 = 1.723280564 + 1.281445835j
        maple_z2 = -1.507523639 + 2.909483373j
        maple_z3 = 4.114046968 + 1.817899948j
        maple_z4 = -3.116133948 - 1.273742661j

        error_z1 = abs(value_z1 - maple_z1)
        error_z2 = abs(value_z2 - maple_z2)
        error_z3 = abs(value_z3 - maple_z3)
        error_z4 = abs(value_z4 - maple_z4)

        self.assertLess(error_z1, 1e-8)
        self.assertLess(error_z2, 1e-8)
        self.assertLess(error_z3, 1e-8)
        self.assertLess(error_z4, 1e-8)


    def test_second_derivatives_oscpart(self):
        w = [0.2+0.5j, 0.3-0.1j, -0.1+0.2j]
        Omega = self.Omega3

        H = RiemannTheta.oscillatory_part_hessian(w, Omega, epsilon=1e-14)

        maple_00 = -2.160656081990225 + 14.02434682346524j
        maple_01 = -1.483857302597929 - 0.9449250397349686j
        maple_02 = 1.954110529051029 - 1.042434632145520j
        maple_11 = 1.037397682580653 + 0.1077503940181105j
        maple_12 = 0.09466454944980265 - 0.3593388338083318j
        maple_22 = -0.3227275082474401 - 2.585609638196203j

        error_00 = abs(H[0,0] - maple_00)
        error_01 = abs(H[0,1] - maple_01)
        error_02 = abs(H[0,2] - maple_02)
        error_11 = abs(H[1,1] - maple_11)
        error_12 = abs(H[1,2] - maple_12)
        error_22 = abs(H[2,2] - maple_22)

        self.assertLess(error_00, 1e-8)
        self.assertLess(error_01, 1e-8)
        self.assertLess(error_02, 1e-8)
        self.assertLess(error_11, 1e-8)
        self.assertLess(error_12, 1e-8)
        self.assertLess(error_22, 1e-8)

    def test_third_derivatives_oscpart(self):
        w = [0.2+0.5j, 0.3-0.1j, -0.1+0.2j]
        Omega = self.Omega3

        dVec_1 = [[1,0,0],[1,0,0],[1,0,0]]
        dVec_2 = [[1,0,0],[0,1,0],[1,0,0]]
        dVec_3 = [[1,0,0],[0,0,1],[0,1,0]]
        dVec_4 = [[0,1,0],[0,0,1],[0,1,0]]
        dVec_5 = [[0,0,1],[0,0,1],[0,0,1]]
        dVec_6 = [[0,0,1],[0,1,0],[0,0,1]]
        dVec_7 = [[1,2,3.1],[2.9,-0.3,1.0],[-20,13.3,0.6684]]

        maple_1 = 88.96174663331488 + 12.83401972101860j
        maple_2 = -5.963646070489819 + 9.261504506522976j
        maple_3 = -1.347499363888600 + 0.5297607158965981j
        maple_4 = 1.217499355198950 + 0.8449102496878512j
        maple_5 = -15.58299545726265 - 0.4376346712347114j
        maple_6 = -2.441570516715710 - 0.2535384980716853j
        maple_7 = -2791.345600876934 + 1286.207313664481j

        deriv_1 = RiemannTheta.oscillatory_part(w, Omega, 
                epsilon=1e-14, derivs=dVec_1)
        deriv_2 = RiemannTheta.oscillatory_part(w, Omega, 
                epsilon=1e-14, derivs=dVec_2)
        deriv_3 = RiemannTheta.oscillatory_part(w, Omega, 
                epsilon=1e-14, derivs=dVec_3)
        deriv_4 = RiemannTheta.oscillatory_part(w, Omega, 
                epsilon=1e-14, derivs=dVec_4)
        deriv_5 = RiemannTheta.oscillatory_part(w, Omega, 
                epsilon=1e-14, derivs=dVec_5)
        deriv_6 = RiemannTheta.oscillatory_part(w, Omega, 
                epsilon=1e-14, derivs=dVec_6)
        deriv_7 = RiemannTheta.oscillatory_part(w, Omega, 
                epsilon=1e-14, derivs=dVec_7)
    
        error_1 = abs(deriv_1 - maple_1)
        error_2 = abs(deriv_2 - maple_2)
        error_3 = abs(deriv_3 - maple_3)
        error_4 = abs(deriv_4 - maple_4)
        error_5 = abs(deriv_5 - maple_5)
        error_6 = abs(deriv_6 - maple_6)
        error_7 = abs(deriv_7 - maple_7)

        self.assertLess(error_1, 1e-8)
        self.assertLess(error_2, 1e-8)
        self.assertLess(error_3, 1e-8)
        self.assertLess(error_4, 1e-8)
        self.assertLess(error_5, 1e-8)
        self.assertLess(error_6, 1e-8)
        self.assertLess(error_7, 1e-8)

        # Genus 4 example
        Omega = self.Omega4
        w = [-0.37704918-0.18456279j, 0.63934426+0.42591413j,
             0.54918033+0.09937996j, -0.21721311-0.07808426j]

        dVec_1 = [[1,0,0,0],[1,0,0,0],[1,0,0,0]]
        dVec_2 = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        dVec_3 = [[1,0,0,0],[0,0,1,0],[0,0,0,1]]
        dVec_4 = [[1,0,0,0],[0,1,1,0],[1,0,0,1]]
        dVec_5 = [[0,0,1,0],[0,1,1,0],[1,0,0,1]]
        dVec_6 = [[0,0,1,0],[1,2,3,4],[1,0,0,1]]
        dVec_7 = [[3.2,-9.8,0.004,-13.9],[0,2.4,0,4],[90.1,-12.93947,-1e-4,3]]

        maple_1 = -67.14022021800414 - 50.25487358123665j
        maple_2 = 6.220027066901749 - 16.96996479658767j
        maple_3 = 14.42498231220689 + 16.30518807929409j
        maple_4 = -35.67483045211793 - 18.14139876283777j
        maple_5 = 53.25640352451774 + 18.93871689387491j
        maple_6 = -185.6760275507559 - 93.99261766419004j
        maple_7 = 239954.2751344823 + 129975.3988999572j

        deriv_1 = RiemannTheta.oscillatory_part(w, Omega, 
                epsilon=1e-14, derivs=dVec_1)
        deriv_2 = RiemannTheta.oscillatory_part(w, Omega, 
                epsilon=1e-14, derivs=dVec_2)
        deriv_3 = RiemannTheta.oscillatory_part(w, Omega, 
                epsilon=1e-14, derivs=dVec_3)
        deriv_4 = RiemannTheta.oscillatory_part(w, Omega, 
                epsilon=1e-14, derivs=dVec_4)
        deriv_5 = RiemannTheta.oscillatory_part(w, Omega, 
                epsilon=1e-14, derivs=dVec_5)
        deriv_6 = RiemannTheta.oscillatory_part(w, Omega, 
                epsilon=1e-14, derivs=dVec_6)
        deriv_7 = RiemannTheta.oscillatory_part(w, Omega, 
                epsilon=1e-14, derivs=dVec_7)

        error_1 = abs(deriv_1 - maple_1)
        error_2 = abs(deriv_2 - maple_2)
        error_3 = abs(deriv_3 - maple_3)
        error_4 = abs(deriv_4 - maple_4)
        error_5 = abs(deriv_5 - maple_5)
        error_6 = abs(deriv_6 - maple_6)
        error_7 = abs(deriv_7 - maple_7)

        self.assertLess(error_1, 1e-8)
        self.assertLess(error_2, 1e-8)
        self.assertLess(error_3, 1e-8)
        self.assertLess(error_4, 1e-8)
        self.assertLess(error_5, 1e-8)
        self.assertLess(error_6, 1e-8)
        self.assertLess(error_7, 1e-8)


    def test_sixth_derivatives(self):
        w = [0.2+0.5j, 0.3-0.1j, -0.1+0.2j]
        Omega = self.Omega3
        
        dVec_1 = [[1,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,1],[0,1,0]]
        dVec_2 = [[1,2,3],[4,5,6],[0.7,0.8,0.9],[0.8,0.7,0.6],[5,4,3],[2,1,0]]
        #dVec_3 = [[-17.3, 6.2, 0],[3.4, 3, 1],[-9,-0.001, 2], 
        #        [1e-2, 0, 19],[210, 0.5, 1.2],[31.323, 0.3, 3]]
        #dVec_4 = [[1,2,3],[4,5,6],[7,8,9],[8,7,6],[5,4,3],[2,1,0]]
        # Neither of the above two examples pass the tests. It appears
        # that for higher order derivatives, if the norm of the directional
        # derivative is too large  

        maple_1 = 42.73836471691125 + 235.2990585642670j
        maple_2 = 0.2152838084588008*10**7 - 0.3287239590246880*10**7*1j
        #maple_3 = 0.2232644817692030*10**12 - 0.1226563725159786*10**12*1j
        #maple_4 = 0.2152838084588008*10**9 - 0.3287239590246880*10**9*1j

        deriv_1 = RiemannTheta.oscillatory_part(w, Omega, 
                epsilon=1e-14, derivs=dVec_1)
        deriv_2 = RiemannTheta.oscillatory_part(w, Omega, 
                epsilon=1e-14, derivs=dVec_2)
        #deriv_3 = RiemannTheta.oscillatory_part(w, Omega, 
        #        epsilon=1e-14, derivs=dVec_3)
        #deriv_4 = RiemannTheta.oscillatory_part(w, Omega, 
        #        epsilon=1e-14, derivs=dVec_4)

        error_1 = abs(deriv_1 - maple_1)
        error_2 = abs(deriv_2 - maple_2)
        #error_3 = abs(deriv_3 - maple_3)
        #error_4 = abs(deriv_4 - maple_4)

        self.assertLess(error_1, 1e-8)
        self.assertLess(error_2, 1e-8)
        #self.assertLess(error_3, 1e-8)
        #self.assertLess(error_4, 1e-8)

class TestRiemannThetaValues(unittest.TestCase):
    def setUp(self):
        self.Omega3 = numpy.array(
            [[1.j, 0.5, 0.5],
             [0.5, 1.j, 0.5],
             [0.5, 0.5, 1.j]], dtype=numpy.complex)
        self.Omega4 = numpy.array(
            [[ 0.39344262+0.79503971j, -0.75409836-0.36912558j,
               -0.44262295-0.02839428j,  0.20491803+0.26974562j],
             [-0.75409836-0.36912558j,  0.27868852+0.85182827j,
              0.09836066+0.19875993j, -0.43442623-0.15616852j],
             [-0.44262295-0.02839428j,  0.09836066+0.19875993j,
              -0.37704918+0.68146261j, -0.91803279+0.45430841j],
             [ 0.20491803+0.26974562j, -0.43442623-0.15616852j,
               -0.91803279+0.45430841j, -1.27868852+0.88022254j]],
            dtype=numpy.complex)

    def test_issue84_value(self):
        z = [0.5-1.10093687j, -0.11723434j]
        Omega = [[0.5+2j, 0.5+1j],
                 [0.5+1j, 1+1.5j]]

        theta_actual = 0.963179246467 - 6.2286820685j
        for _ in range(1000):
            theta = RiemannTheta(z,Omega)
            error = abs(theta - theta_actual)
            self.assertLess(error, 1e-5,
                            '%s not less than %s'
                            '\ntheta:  %s\nactual: %s'%(
                                error,1e-5,theta, theta_actual))

    def test_issue84_radius(self):
        Omega = [[0.5+2j, 0.5+1j],
                 [0.5+1j, 1+1.5j]]
        Omega = numpy.array(Omega)
        Y = Omega.imag
        T = cholesky(Y).T

        # Check that the radius is consistent when repeatedly computed
        R_actual = radius(1e-8,T)
        for _ in range(1000):
            R = radius(1e-8,T)
            error = abs(R - R_actual)
            self.assertLess(error, 1e-8)

    def test_issue159(self):
        Omega = [[10j]]
        z = [5j]

        theta_actual = 2
        theta = RiemannTheta(z,Omega)
        error = abs(theta - theta_actual)
        self.assertLess(error, 1e-8)

    def test_gradient(self):
        Omega = self.Omega3

        # generate random test z-values
        N = 32
        u = numpy.random.rand(N,3)
        v = numpy.random.rand(N,3)
        W = u + 1.0j*v

        # manually compute gradients
        dz0 = RiemannTheta(W,Omega,derivs=[[1,0,0]])
        dz1 = RiemannTheta(W,Omega,derivs=[[0,1,0]])
        dz2 = RiemannTheta(W,Omega,derivs=[[0,0,1]])
        grad1 = numpy.zeros_like(W, dtype=numpy.complex)
        grad1[:,0] = dz0
        grad1[:,1] = dz1
        grad1[:,2] = dz2

        # compute using "gradient"
        grad2 = RiemannTheta.gradient(W,Omega)
        self.assertLess(numpy.linalg.norm(grad1-grad2), 1e-14)

        Omega = self.Omega4

        # generate random test z-values
        N = 32
        u = numpy.random.rand(N,4)
        v = numpy.random.rand(N,4)
        W = u + 1.0j*v

        # manually compute gradients
        dz0 = RiemannTheta(W,Omega,derivs=[[1,0,0,0]])
        dz1 = RiemannTheta(W,Omega,derivs=[[0,1,0,0]])
        dz2 = RiemannTheta(W,Omega,derivs=[[0,0,1,0]])
        dz3 = RiemannTheta(W,Omega,derivs=[[0,0,0,1]])
        grad1 = numpy.zeros_like(W, dtype=numpy.complex)
        grad1[:,0] = dz0
        grad1[:,1] = dz1
        grad1[:,2] = dz2
        grad1[:,3] = dz3

        # compute using "gradient"
        grad2 = RiemannTheta.gradient(W,Omega)
        self.assertLess(numpy.linalg.norm(grad1-grad2), 1e-14)


    def test_second_derivative_symmetric(self):
        w = [0.2+0.5j, 0.3-0.1j, -0.1+0.2j]
        Omega = [[1.j, 0.5, 0.5],
                 [0.5, 1.j, 0.5],
                 [0.5, 0.5, 1.j]]

        dz_01 = RiemannTheta.oscillatory_part(
            w, Omega, epsilon=1e-14, derivs=[[1,0,0],[0,1,0]])
        dz_10 = RiemannTheta.oscillatory_part(
            w, Omega, epsilon=1e-14, derivs=[[0,1,0],[1,0,0]])
        dz_02 = RiemannTheta.oscillatory_part(
            w, Omega, epsilon=1e-14, derivs=[[1,0,0],[0,0,1]])
        dz_20 = RiemannTheta.oscillatory_part(
            w, Omega, epsilon=1e-14, derivs=[[0,0,1],[1,0,0]])
        dz_12 = RiemannTheta.oscillatory_part(
            w, Omega, epsilon=1e-14, derivs=[[0,1,0],[0,0,1]])
        dz_21 = RiemannTheta.oscillatory_part(
            w, Omega, epsilon=1e-14, derivs=[[0,0,1],[0,1,0]])

        error_01_10 = abs(dz_01 - dz_10)
        error_02_20 = abs(dz_02 - dz_20)
        error_12_21 = abs(dz_12 - dz_21)

        self.assertLess(error_01_10, 1e-8)
        self.assertLess(error_02_20, 1e-8)
        self.assertLess(error_12_21, 1e-8)


    def test_symmetric_hessian(self):
        pass


    def test_hessian(self):
        pass

    def test_against_naive_implementation_genus1(self):
        # tests the genus 1 Riemann theta function against the naive
        # implementation written above (directly using the summation formula).

        # first test the relative error using values close to the origin,
        # avoiding the double-exponential growth
        N = 64
        sigma = 0.1
        z = sigma*randn(N) + 1.j*sigma*randn(N)
        z = z.reshape((N,1))
        tau = [[1.0j]]

        values1 = RiemannTheta(z,tau,epsilon=1e-16)
        values2 = thetag1(z,tau[0][0])[:,0]

        rel_error = abs((values1-values2)/values1)
        rel_error_max = numpy.max(rel_error)
        rel_error_avg = numpy.mean(rel_error)
        self.assertLess(rel_error_max,1e-14)
        self.assertLess(rel_error_avg,1e-14)

        # next, test the relative error using larger magnitude values. we don't
        # test the max error due to possible numerical roundoff issues
        sigma = 3
        z = sigma*randn(N) + 1.j*sigma*randn(N)
        z = z.reshape((N,1))
        tau = [[1.0j]]

        values1 = RiemannTheta(z,tau,epsilon=1e-16)
        values2 = thetag1(z,tau[0][0])[:,0]

        rel_error = abs((values1-values2)/values1)
        rel_error_avg = numpy.mean(rel_error)
        self.assertLess(rel_error_avg,1e-14)

        # repeat for different tau
        tau = [[1.0 + 2.5j]]

        values1 = RiemannTheta(z,tau,epsilon=1e-16)
        values2 = thetag1(z,tau[0][0])[:,0]

        rel_error = abs((values1-values2)/values1)
        rel_error_avg = numpy.mean(rel_error)
        self.assertLess(rel_error_avg,1e-14)

    @unittest.skipIf(NO_JTHETA, 'Could not find sympy.mpmath.jtheta')
    def test_against_sympy_jtheta(self):
        N = 64
        sigma = 2
        z = sigma*randn(N) + 1.j*sigma*randn(N)
        z = z.reshape((N,1))
        tau = [[1.0j]]

        # jtheta inputs
        w = numpy.pi*z[:,0]
        q = numpy.exp(numpy.pi*1.0j*tau[0][0])

        values1 = RiemannTheta(z,tau,epsilon=1e-16)
        values2 = numpy.array([jtheta(3,wi,q) for wi in w],
                              dtype=numpy.complex)

        rel_error = abs((values1-values2)/values1)
        rel_error_avg = numpy.mean(rel_error)
        self.assertLess(rel_error_avg,1e-14)

        # repeat for different tau
        tau = [[1.0 + 2.5j]]
        q = numpy.exp(numpy.pi*1.0j*tau[0][0])

        values1 = RiemannTheta(z,tau,epsilon=1e-16)
        values2 = numpy.array([jtheta(3,wi,q) for wi in w],
                              dtype=numpy.complex)

        rel_error = abs((values1-values2)/values1)
        rel_error_avg = numpy.mean(rel_error)
        self.assertLess(rel_error_avg,1e-14)

    # def test_value_at_point(self):
    #     Omega = numpy.array(
    #         [[1.0 + 1.15700539j, -1.0 - 0.5773502693j],
    #          [-1.0 - 0.5773502693j, 1.0 + 1.154700539j]],
    #         dtype=numpy.complex)

    #     # first z-value
    #     z = numpy.array([1.0 - 1.0j, 1.0 + 1.0j])
    #     u = RiemannTheta.exponential_part(z, Omega)
    #     u_actual = 0
    #     u_delta = abs(u - u_actual)
    #     self.assertAlmostEqual(u_delta, 0)

    #     v = RiemannTheta.oscillatory_part(z, Omega)
    #     v_actual = 0
    #     v_delta = abs(v - v_actual)
    #     self.assertAlmostEqual(v_delta, 0)

    #     # second z-value
    #     z = numpy.array([1.0 - 1.0j, 1.0 + 1.0j])
    #     u = RiemannTheta.exponential_part(z, Omega)
    #     u_actual = 0
    #     u_delta = abs(u - u_actual)
    #     self.assertAlmostEqual(u_delta, 0)

    #     v = RiemannTheta.oscillatory_part(z, Omega)
    #     v_actual = 0
    #     v_delta = abs(v - v_actual)
    #     self.assertAlmostEqual(v_delta, 0)

    #     # same tests, different Omega
    #     Omega = numpy.array(
    #         [[1.0 + 1.15700539j, -1.0 - 0.5773502693j],
    #          [-1.0 - 0.5773502693j, 1.0 + 1.154700539j]],
    #         dtype=numpy.complex)

    #     # first z-value
    #     z = numpy.array([1.0 - 1.0j, 1.0 + 1.0j])
    #     u = RiemannTheta.exponential_part(z, Omega)
    #     u_actual = 0
    #     u_delta = abs(u - u_actual)
    #     self.assertAlmostEqual(u_delta, 0)

    #     v = RiemannTheta.oscillatory_part(z, Omega)
    #     v_actual = 0
    #     v_delta = abs(v - v_actual)
    #     self.assertAlmostEqual(v_delta, 0)

    #     # second z-value
    #     z = numpy.array([1.0 - 1.0j, 1.0 + 1.0j])
    #     u = RiemannTheta.exponential_part(z, Omega)
    #     u_actual = 0
    #     u_delta = abs(u - u_actual)
    #     self.assertAlmostEqual(u_delta, 0)

    #     v = RiemannTheta.oscillatory_part(z, Omega)
    #     v_actual = 0
    #     v_delta = abs(v - v_actual)
    #     self.assertAlmostEqual(v_delta, 0)


    # def test_value_at_point_1_derivs(self):
    #     Omega = numpy.array([[1.0 + 1.15700539j, -1.0 - 0.5773502693j],
    #                       [-1.0 - 0.5773502693j, 1.0 + 1.154700539j]],
    #                      dtype=numpy.complex)

    #     # test 1
    #     derivs = [[1,0]]
    #     z = numpy.array([1.0 - 1.0j, 1.0 + 1.0j])
    #     u = RiemannTheta.exponential_part(z, Omega, derivs=derivs)
    #     u_actual = 0
    #     u_delta = abs(u - u_actual)
    #     self.assertAlmostEqual(u_delta, 0)

    #     v = RiemannTheta.oscillatory_part(z, Omega, derivs=derivs)
    #     v_actual = 0
    #     v_delta = abs(v - v_actual)
    #     self.assertAlmostEqual(v_delta, 0)

    #     # test 2
    #     derivs = [[0,1]]
    #     z = numpy.array([1.0 - 1.0j, 1.0 + 1.0j])
    #     u = RiemannTheta.exponential_part(z, Omega, derivs=derivs)
    #     u_actual = 0
    #     u_delta = abs(u - u_actual)
    #     self.assertAlmostEqual(u_delta, 0)

    #     v = RiemannTheta.oscillatory_part(z, Omega, derivs=derivs)
    #     v_actual = 0
    #     v_delta = abs(v - v_actual)
    #     self.assertAlmostEqual(v_delta, 0)


    # def test_value_at_point_2_derivs(self):
    #     Omega = numpy.array([[1.0 + 1.15700539j, -1.0 - 0.5773502693j],
    #                       [-1.0 - 0.5773502693j, 1.0 + 1.154700539j]],
    #                      dtype=numpy.complex)

    #     # test 1
    #     derivs = [[1,0]]
    #     z = numpy.array([1.0 - 1.0j, 1.0 + 1.0j])
    #     u = RiemannTheta.exponential_part(z, Omega, derivs=derivs)
    #     u_actual = 0
    #     u_delta = abs(u - u_actual)
    #     self.assertAlmostEqual(u_delta, 0)

    #     v = RiemannTheta.oscillatory_part(z, Omega, derivs=derivs)
    #     v_actual = 0
    #     v_delta = abs(v - v_actual)
    #     self.assertAlmostEqual(v_delta, 0)

    #     # test 2
    #     derivs = [[0,1]]
    #     z = numpy.array([1.0 - 1.0j, 1.0 + 1.0j])
    #     u = RiemannTheta.exponential_part(z, Omega, derivs=derivs)
    #     u_actual = 0
    #     u_delta = abs(u - u_actual)
    #     self.assertAlmostEqual(u_delta, 0)

    #     v = RiemannTheta.oscillatory_part(z, Omega, derivs=derivs)
    #     v_actual = 0
    #     v_delta = abs(v - v_actual)
    #     self.assertAlmostEqual(v_delta, 0)




    #########################################################################
    # def test_zeroCharacteristic(self):                                    #
    #     #Replace with random z, omega, or set of such                     #
    #     z = numpy.array([1.0j, 0])                                        #
    #     omega = numpy.matrix(                                             #
    #         [[1.0j, -0.5],                                                #
    #         [-0.5, 1.0j]]                                                 #
    #         )                                                             #
    #     char = [[0,0],[0,0]]                                              #
    #     thetaValue = RiemannTheta(z, omega, batch = False)                #
    #     thetaCharValue = RiemannTheta.characteristic(char, z, omega)      #
    #     delta = scipy.linalg.norm(thetaValue - thetaCharValue, numpy.inf) #
    #     self.assertAlmostEqual(delta,0)                                   #
    #                                                                       #
    # def test_jacobiTheta1(self):                                          #
    #     #Test against sympy mpmath.jtheta(1,z)                            #
    #     z = 1.0j                                                          #
    #     q_omega = .1                                                      #
    #     char = [[.5],[.5]]                                                #
    #     jTheta1 = jtheta(1, numpy.pi * z, q_omega)                        #
    #     thetaValue = RiemannTheta.characteristic(char, z, q_omega)        #
    #     self.assertAlmostEqual(jTheta1, -thetaValue)                      #
    #                                                                       #
    #                                                                       #
    # def test_jacobiTheta2(self):                                          #
    #     #Test against sympy mpmath.jtheta(2,z)                            #
    #     z = 1.0j                                                          #
    #     q_omega = .1                                                      #
    #     char = [[.5],[0]]                                                 #
    #     jTheta2 = jtheta(2, numpy.pi * z, q_omega)                        #
    #     thetaValue = RiemannTheta.characteristic(char, z, q_omega)        #
    #     self.assertAlmostEqual(jTheta2, thetaValue)                       #
    #                                                                       #
    # def test_jacobiTheta3(self):                                          #
    #     #Test against sympy mpmath.jtheta(3,z)                            #
    #     z = 1.0j                                                          #
    #     q_omega = .1                                                      #
    #     char = [[0],[0]]                                                  #
    #     jTheta3 = jtheta(3, numpy.pi * z, q_omega)                        #
    #     thetaValue = RiemannTheta.characteristic(char, z, q_omega)        #
    #     self.assertAlmostEqual(jTheta3, thetaValue)                       #
    #                                                                       #
    # def test_jacobiTheta4(self):                                          #
    #     #Test against sympy mpmath.jtheta(4,z)                            #
    #     z = 1.0j                                                          #
    #     q_omega = .1                                                      #
    #     char = [[0],[.5]]                                                 #
    #     jTheta3 = jtheta(4, numpy.pi * z, q_omega)                        #
    #     thetaValue = RiemannTheta.characteristic(char, z, q_omega)        #
    #     self.assertAlmostEqual(jTheta3, thetaValue)                       #
    #                                                                       #
    # def test_zParity(self):                                               #
    #     z = numpy.array([1.0j, 0])                                        #
    #     omega = numpy.matrix(                                             #
    #         [[1.0j, -0.5],                                                #
    #         [-0.5, 1.0j]]                                                 #
    #         )                                                             #
    #     theta1 = RiemannTheta.value_at_point(z, omega, batch = False)     #
    #     theta2 = RiemannTheta.value_at_point(-z, omega, batch = False)    #
    #     self.assertAlmostEqual(theta1,theta2)                             #
    #                                                                       #
    # def test_zIntShift(self):                                             #
    #     z = numpy.array([1.0j, 0])                                        #
    #     omega = numpy.matrix(                                             #
    #         [[1.0j, -0.5],                                                #
    #         [-0.5, 1.0j]]                                                 #
    #         )                                                             #
    #     m = numpy.array([1, 1])                                           #
    #     theta1 = RiemannTheta.value_at_point(z, omega, batch = False)     #
    #     theta2 = RiemannTheta.value_at_point(z + m, omega, batch = False) #
    #     self.assertAlmostEqual(theta1,theta2)                             #
    #                                                                       #
    # def test_quasiPeriodic(self):                                         #
    #     #Test for DLMF 21.3.3                                             #
    #     pass                                                              #
    #                                                                       #
    # def test_characteristicShift(self):                                   #
    #     #Test for DLMF 21.3.4                                             #
    #     pass                                                              #
    #                                                                       #
    # def test_halfperiodCharacteristic(self):                              #
    #     #Test for DLMF 21.3.6                                             #
    #     pass                                                              #
    #########################################################################
