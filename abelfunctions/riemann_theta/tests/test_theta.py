import unittest
import numpy as np
import scipy
#Look further into mpmath documentation
from sympy.mpmath import jtheta

from abelfunctions.riemanntheta import RiemannTheta

""" 
References
----------
.. [CRTF] Computing Riemann Theta Functions. Bernard Deconinck, Matthias 
   Heil, Alexander Bobenko, Mark van Hoeij and Markus Schmies.  
   Mathematics of Computation 73 (2004) 1417-1442.  Paper available at
   http://depts.washington.edu/bdecon/papers/pdfs/computingtheta.pdf   
   Accompanying Maple code available at 
   http://www.math.fsu.edu/~hoeij/RiemannTheta/
.. [DLMF] Digital Library of Mathematics Functions - Riemann Theta 
   Functions ( http://dlmf.nist.gov/21 ).
   
.. [SAGE] Computing Riemann theta functions in Sage with applications.
   Christopher Swierczewski and Bernard Deconinck.  Submitted for 
   publication.  Available online at 
   http://depts.washington.edu/bdecon/papers/pdfs/Swierczewski_Deconinck1.pdf
"""

class RiemannThetaValueTest(unittest.TestCase):
    
    def setup(self):
        self.theta = RiemannTheta_Function()    
   
    @unittest.expectedFailure   
    def test_valueAtPoint1(self):        
        z = np.array([1.0 - 1.0j, 1.0 + 1.0j])
        omega = np.matrix(
           [[1.0 + 1.15700539j, -1.0 - 0.5773502693j],
           [-1.0 - 0.5773502693j, 1.0 + 1.154700539j]])
        thetaValue = theta.value_at_point(z, omega, batch = False)
        thetaSoln = np.array([-21.76547256 - 0.232329832610e-8j])
        delta = scipy.linalg.norm(thetaValue - thetaSoln, np.inf)
        self.assertAlmostEqual(delta, 0, 'broken')        
        
    def test_valueAtPoint2(self):
        z = np.array([0, 0])
        omega = np.matrix(
           [[1.0j, -0.5],
           [-0.5, 1.0j]])
        thetaValue = theta.value_at_point(z, omega, batch = False)
        thetaSoln = np.array([1.16540106 - 6.42668155e-24j])
        delta = scipy.linalg.norm(thetaValue - thetaSoln, np.inf)
        self.assertAlmostEqual(delta, 0)        

    def test_valueAtPoint3(self):
        z = np.array([.5, 1/3. + .25j])
        omega = np.matrix(
           [[1.0j, -0.5],
           [-0.5, 1.0j]])
        thetaValue = theta.value_at_point(z, omega, batch = False)
        thetaSoln = np.array([.795738522 - 0.187073837j])
        delta = scipy.linalg.norm(thetaValue - thetaSoln, np.inf)
        self.assertAlmostEqual(delta, 0)        
    
#    def test_batchValueAtPoint(self):
#        z0 = np.array([0, 0])
#        z1 = np.array([1.0j,1.0j])
#        z2 = np.array([.5 + .5j, .5 + .5j])
#        z3 = np.array([0 + .5j, .33 + .8j])
#        z4 = np.array([.345 + .768j, -44 - .76j])
#        omega = np.matrix(
#           [[1.0j, -0.5],
#           [-0.5, 1.0j]])
#        thetaValue = theta.value_at_point(
#            [z0, z1, z2, z3, z4], omega, batch = True)
#        Need values to assert against
#        thetaSoln = 
#        delta = 
#        self.assertAlmostEqual    
    
    @unittest.expectedFailure
    def test_derivativesTheta1(self):
        z = np.array([1.0j, 0])
        omega = np.matrix(
           [[1.0j, -0.5],
           [-0.5, 1.0j]])
        thetaValue = theta.value_at_point(z, omega, deriv = [[1,0]])
        thetaSoln = np.array([0.0 - 146.49j])
        delta = scipy.linalg.norm(thetaValue - thetaSoln, np.inf)
        self.assertAlmostEqual(delta, 0)        
    
    def test_derivativesTheta2(self):
        z = np.array([1.0j, 0])
        omega = np.matrix(
            [[1.0j, -0.5],
            [-0.5, 1.0j]]
            )
        thetaValue = theta.value_at_point(z, omega, deriv = [[1,0],[0,1]])
        thetaSoln = np.array([0.0 - 0j])
        delta = scipy.linalg.norm(thetaValue - thetaSoln, np.inf)
        self.assertAlmostEqual(delta, 0)    
    
    def test_derivativesTheta3(self):
        z = np.array([1.0j, 0])
        omega = np.matrix(
            [[1.0j, -0.5],
            [-0.5, 1.0j]]
            )
        thetaValue = theta.value_at_point(z, omega, deriv = [[0,1],[1,0]])
        thetaSoln = np.array([0.0 - 0j])
        delta = scipy.linalg.norm(thetaValue - thetaSoln, np.inf)
        self.assertAlmostEqual(delta, 0)    
    
    @unittest.expectedFailure
    def test_derivativesTheta4(self):
        z = np.array([1.0j, 0])
        omega = np.matrix(
            [[1.0j, -0.5],
            [-0.5, 1.0j]]
            )
        thetaValue = theta.value_at_point(
            z, omega, deriv = [[1,0],[1,0],[1,1]])
        thetaSoln = np.array([0.0 - 7400.39j])
        delta = scipy.linalg.norm(thetaValue - thetaSoln, np.inf)
        self.assertAlmostEqual(delta, 0)        
        
    @unittest.expectedFailure
    def test_derivativesTheta5(self):
        z = np.array([1.0j, 0])
        omega = np.matrix(
            [[1.0j, -0.5],
            [-0.5, 1.0j]]
            )
        thetaValue = theta.value_at_point(
            z, omega, deriv = [[1,1],[1,1],[1,1],[1,1]])
        thetaSoln = np.array([0.0 - 7400.39j])
        delta = scipy.linalg.norm(thetaValue - thetaSoln, np.inf)
        self.assertAlmostEqual(delta, 0)     
     
#    def test_characteristicTheta(self):
#        z = np.array([1.0j,0])
#        omega = np.matrix([
#            [1.0j,-0.5],
#            [-0.5,1.0j]
#            ])
#        deriv = [[1,0],[1,0]]
#        chars = [[0,0],[0,0]]    
     
class RiemannThetaTest(unittest.TestCase):
    
    """
    Note: z and omega should probably be randomly generated in batches
    for comprehensive testing
    """
    
    def setup(self):
        self.theta = RiemannTheta_Function() 
    
    def test_zeroCharacteristic(self):
        #Replace with random z, omega, or set of such
        z = np.array([1.0j, 0])
        omega = np.matrix(
            [[1.0j, -0.5],
            [-0.5, 1.0j]]
            )
        char = [[0,0],[0,0]]
        thetaValue = theta.value_at_point(z, omega, batch = False)
        thetaCharValue = theta.characteristic(char, z, omega)
        delta = scipy.linalg.norm(thetaValue - thetaCharValue, np.inf)
        self.assertAlmostEqual(delta,0)
    
    def test_jacobiTheta1(self):
        #Test against sympy mpmath.jtheta(1,z)       
        z = 1.0j
        q_omega = .1
        char = [[.5],[.5]]
        jTheta1 = jtheta(1, np.pi * z, q_omega)
        thetaValue = theta.characteristic(char, z, q_omega)        
        self.assertAlmostEqual(jTheta1, -thetaValue)

    
    def test_jacobiTheta2(self):
        #Test against sympy mpmath.jtheta(2,z)
        z = 1.0j
        q_omega = .1
        char = [[.5],[0]]
        jTheta2 = jtheta(2, np.pi * z, q_omega)
        thetaValue = theta.characteristic(char, z, q_omega)        
        self.assertAlmostEqual(jTheta2, thetaValue)   
    
    def test_jacobiTheta3(self):
        #Test against sympy mpmath.jtheta(3,z)
        z = 1.0j
        q_omega = .1
        char = [[0],[0]]
        jTheta3 = jtheta(3, np.pi * z, q_omega)
        thetaValue = theta.characteristic(char, z, q_omega)        
        self.assertAlmostEqual(jTheta3, thetaValue)
    
    def test_jacobiTheta4(self):
        #Test against sympy mpmath.jtheta(4,z)
        z = 1.0j
        q_omega = .1
        char = [[0],[.5]]
        jTheta3 = jtheta(4, np.pi * z, q_omega)
        thetaValue = theta.characteristic(char, z, q_omega)        
        self.assertAlmostEqual(jTheta3, thetaValue)
        
    def test_zParity(self):
        z = np.array([1.0j, 0])
        omega = np.matrix(
            [[1.0j, -0.5],
            [-0.5, 1.0j]]
            )
        theta1 = theta.value_at_point(z, omega, batch = False)
        theta2 = theta.value_at_point(-z, omega, batch = False)
        self.assertAlmostEqual(theta1,theta2)
    
    def test_zIntShift(self):
        z = np.array([1.0j, 0])
        omega = np.matrix(
            [[1.0j, -0.5],
            [-0.5, 1.0j]]
            )
        m = np.array([1, 1])
        theta1 = theta.value_at_point(z, omega, batch = False)
        theta2 = theta.value_at_point(z + m, omega, batch = False)
        self.assertAlmostEqual(theta1,theta2)
        
    def test_quasiPeriodic(self):
        #Test for DLMF 21.3.3
        pass
    
    def test_characteristicShift(self):
        #Test for DLMF 21.3.4
        pass
    
    def test_halfperiodCharacteristic(self):
        #Test for DLMF 21.3.6
        
        
if __name__ == '__main__':
    unittest.main()