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
        self.Omega1 = numpy.array(
            [[1.j, 0.5, 0.5],
             [0.5, 1.j, 0.5],
             [0.5, 0.5, 1.j]], dtype=numpy.complex)

    def test_value(self):
        z = [0,0,0]
        Omega = self.Omega1

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
        Omega = self.Omega1

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


    def test_second_derivatives(self):
        w = [0.2+0.5j, 0.3-0.1j, -0.1+0.2j]
        Omega = self.Omega1

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


class TestRiemannThetaValues(unittest.TestCase):
    def setup(self):
        pass

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

        R_actual = 5.01708695504
        for _ in range(1000):
            R = radius(1e-8,T)
            error = abs(R - R_actual)
            self.assertLess(error, 1e-8)

    def test_gradient(self):
        Omega = [[1.j, 0.5, 0.5],
                 [0.5, 1.j, 0.5],
                 [0.5, 0.5, 1.j]]

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
    #     Omega = np.array([[1.0 + 1.15700539j, -1.0 - 0.5773502693j],
    #                       [-1.0 - 0.5773502693j, 1.0 + 1.154700539j]],
    #                      dtype=numpy.complex)

    #     # first z-value
    #     z = np.array([1.0 - 1.0j, 1.0 + 1.0j])
    #     u = RiemannTheta.exponential_part(z, Omega)
    #     u_actual = 0
    #     u_delta = abs(u - u_actual)
    #     self.assertAlmostEqual(u_delta, 0)

    #     v = RiemannTheta.oscillatory_part(z, Omega)
    #     v_actual = 0
    #     v_delta = abs(v - v_actual)
    #     self.assertAlmostEqual(v_delta, 0)

    #     # second z-value
    #     z = np.array([1.0 - 1.0j, 1.0 + 1.0j])
    #     u = RiemannTheta.exponential_part(z, Omega)
    #     u_actual = 0
    #     u_delta = abs(u - u_actual)
    #     self.assertAlmostEqual(u_delta, 0)

    #     v = RiemannTheta.oscillatory_part(z, Omega)
    #     v_actual = 0
    #     v_delta = abs(v - v_actual)
    #     self.assertAlmostEqual(v_delta, 0)

    #     # same tests, different Omega
    #     Omega = np.array([[1.0 + 1.15700539j, -1.0 - 0.5773502693j],
    #                       [-1.0 - 0.5773502693j, 1.0 + 1.154700539j]],
    #                      dtype=numpy.complex)

    #     # first z-value
    #     z = np.array([1.0 - 1.0j, 1.0 + 1.0j])
    #     u = RiemannTheta.exponential_part(z, Omega)
    #     u_actual = 0
    #     u_delta = abs(u - u_actual)
    #     self.assertAlmostEqual(u_delta, 0)

    #     v = RiemannTheta.oscillatory_part(z, Omega)
    #     v_actual = 0
    #     v_delta = abs(v - v_actual)
    #     self.assertAlmostEqual(v_delta, 0)

    #     # second z-value
    #     z = np.array([1.0 - 1.0j, 1.0 + 1.0j])
    #     u = RiemannTheta.exponential_part(z, Omega)
    #     u_actual = 0
    #     u_delta = abs(u - u_actual)
    #     self.assertAlmostEqual(u_delta, 0)

    #     v = RiemannTheta.oscillatory_part(z, Omega)
    #     v_actual = 0
    #     v_delta = abs(v - v_actual)
    #     self.assertAlmostEqual(v_delta, 0)


    # def test_value_at_point_1_derivs(self):
    #     Omega = np.array([[1.0 + 1.15700539j, -1.0 - 0.5773502693j],
    #                       [-1.0 - 0.5773502693j, 1.0 + 1.154700539j]],
    #                      dtype=numpy.complex)

    #     # test 1
    #     derivs = [[1,0]]
    #     z = np.array([1.0 - 1.0j, 1.0 + 1.0j])
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
    #     z = np.array([1.0 - 1.0j, 1.0 + 1.0j])
    #     u = RiemannTheta.exponential_part(z, Omega, derivs=derivs)
    #     u_actual = 0
    #     u_delta = abs(u - u_actual)
    #     self.assertAlmostEqual(u_delta, 0)

    #     v = RiemannTheta.oscillatory_part(z, Omega, derivs=derivs)
    #     v_actual = 0
    #     v_delta = abs(v - v_actual)
    #     self.assertAlmostEqual(v_delta, 0)


    # def test_value_at_point_2_derivs(self):
    #     Omega = np.array([[1.0 + 1.15700539j, -1.0 - 0.5773502693j],
    #                       [-1.0 - 0.5773502693j, 1.0 + 1.154700539j]],
    #                      dtype=numpy.complex)

    #     # test 1
    #     derivs = [[1,0]]
    #     z = np.array([1.0 - 1.0j, 1.0 + 1.0j])
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
    #     z = np.array([1.0 - 1.0j, 1.0 + 1.0j])
    #     u = RiemannTheta.exponential_part(z, Omega, derivs=derivs)
    #     u_actual = 0
    #     u_delta = abs(u - u_actual)
    #     self.assertAlmostEqual(u_delta, 0)

    #     v = RiemannTheta.oscillatory_part(z, Omega, derivs=derivs)
    #     v_actual = 0
    #     v_delta = abs(v - v_actual)
    #     self.assertAlmostEqual(v_delta, 0)


    # def test_zeroCharacteristic(self):
    #     #Replace with random z, omega, or set of such
    #     z = np.array([1.0j, 0])
    #     omega = np.matrix(
    #         [[1.0j, -0.5],
    #         [-0.5, 1.0j]]
    #         )
    #     char = [[0,0],[0,0]]
    #     thetaValue = theta.value_at_point(z, omega, batch = False)
    #     thetaCharValue = theta.characteristic(char, z, omega)
    #     delta = scipy.linalg.norm(thetaValue - thetaCharValue, np.inf)
    #     self.assertAlmostEqual(delta,0)

    # def test_jacobiTheta1(self):
    #     #Test against sympy mpmath.jtheta(1,z)
    #     z = 1.0j
    #     q_omega = .1
    #     char = [[.5],[.5]]
    #     jTheta1 = jtheta(1, np.pi * z, q_omega)
    #     thetaValue = theta.characteristic(char, z, q_omega)
    #     self.assertAlmostEqual(jTheta1, -thetaValue)


    # def test_jacobiTheta2(self):
    #     #Test against sympy mpmath.jtheta(2,z)
    #     z = 1.0j
    #     q_omega = .1
    #     char = [[.5],[0]]
    #     jTheta2 = jtheta(2, np.pi * z, q_omega)
    #     thetaValue = theta.characteristic(char, z, q_omega)
    #     self.assertAlmostEqual(jTheta2, thetaValue)

    # def test_jacobiTheta3(self):
    #     #Test against sympy mpmath.jtheta(3,z)
    #     z = 1.0j
    #     q_omega = .1
    #     char = [[0],[0]]
    #     jTheta3 = jtheta(3, np.pi * z, q_omega)
    #     thetaValue = theta.characteristic(char, z, q_omega)
    #     self.assertAlmostEqual(jTheta3, thetaValue)

    # def test_jacobiTheta4(self):
    #     #Test against sympy mpmath.jtheta(4,z)
    #     z = 1.0j
    #     q_omega = .1
    #     char = [[0],[.5]]
    #     jTheta3 = jtheta(4, np.pi * z, q_omega)
    #     thetaValue = theta.characteristic(char, z, q_omega)
    #     self.assertAlmostEqual(jTheta3, thetaValue)

    # def test_zParity(self):
    #     z = np.array([1.0j, 0])
    #     omega = np.matrix(
    #         [[1.0j, -0.5],
    #         [-0.5, 1.0j]]
    #         )
    #     theta1 = theta.value_at_point(z, omega, batch = False)
    #     theta2 = theta.value_at_point(-z, omega, batch = False)
    #     self.assertAlmostEqual(theta1,theta2)

    # def test_zIntShift(self):
    #     z = np.array([1.0j, 0])
    #     omega = np.matrix(
    #         [[1.0j, -0.5],
    #         [-0.5, 1.0j]]
    #         )
    #     m = np.array([1, 1])
    #     theta1 = theta.value_at_point(z, omega, batch = False)
    #     theta2 = theta.value_at_point(z + m, omega, batch = False)
    #     self.assertAlmostEqual(theta1,theta2)

    # def test_quasiPeriodic(self):
    #     #Test for DLMF 21.3.3
    #     pass

    # def test_characteristicShift(self):
    #     #Test for DLMF 21.3.4
    #     pass

    # def test_halfperiodCharacteristic(self):
    #     #Test for DLMF 21.3.6
