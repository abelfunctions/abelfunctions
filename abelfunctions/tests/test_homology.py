import unittest

from sage.all import (
    GF, ZZ, QQ, CDF, RDF, I, Matrix, zero_matrix, identity_matrix, real, imag
)
from abelfunctions.homology import (
    involution_matrix,
    integer_kernel_basis,
    N1_matrix,
    symmetric_block_diagonalize,
    diagonal_locations,
    symmetric_transformation_matrix,
    symmetrize_periods,
)

class HomologyTestData(unittest.TestCase):
    # this class only defines the test data from Kalla and Klein
    def setUp(self):
        atrott = Matrix(
            CDF,
            [[ 0.0000 + 0.0235*I, 0.0000 + 0.0138*I, 0.0000 + 0.0138*I],
             [ 0.0000 + 0.0000*I, 0.0000 + 0.0277*I, 0.0000 + 0.0000*I],
             [-0.0315 + 0.0000*I, 0.0000 + 0.0000*I, 0.0250 + 0.0000*I]])
        btrott = Matrix(
            CDF,
            [[-0.0315 + 0.0235*I, 0.0000 + 0.0138*I,-0.0250 + 0.0138*I],
             [ 0.0000 + 0.0000*I,-0.0250 + 0.0277*I, 0.0250 + 0.0000*I],
             [ 0.0000 - 0.0235*I, 0.0000 + 0.0138*I, 0.0000 + 0.0138*I]])

        # klein curve:
        aklein = Matrix(
            CDF,
            [[-0.9667 + 0.7709*I, 0.9667 + 0.2206*I, 0.9667 - 2.0073*I],
             [-1.2054 - 0.2751*I,-0.4302 + 0.8933*I,-1.7419 + 1.3891*I],
             [-0.4302 - 0.8933*I, 1.7419 + 1.3891*I,-1.2054 + 0.2751*I]])
        bklein = Matrix(
            CDF,
            [[-2.7085 - 0.6182*I,-0.2387 + 0.4958*I, 1.3969 - 1.1140*I],
             [-2.1721 - 1.7322*I, 0.5365 - 0.1224*I,-0.7753 - 1.6097*I],
             [ 0.9667 + 0.2206*I,-0.9667 + 2.0073*I,-0.9667 + 0.7709*I]])

        # fermat curve:
        afermat = Matrix(
            CDF,
            [[0.9270 + 0.0000*I, 0.0000 - 0.9270*I, 0.0000 - 0.9270*I],
             [0.0000 + 0.0000*I, 0.0000 + 0.0000*I, 0.0000 - 1.8541*I],
             [0.0000 + 0.9270*I,-0.9270 + 0.0000*I, 0.0000 - 0.9270*I]])
        bfermat = Matrix(
            CDF,
            [[0.9270 + 0.9270*I, 0.9270 - 0.9270*I, 0.0000 + 0.0000*I],
             [0.0000 + 0.0000*I,-0.9270 + 0.9270*I, 0.9270 - 0.9270*I],
             [-0.9270+ 0.0000*I, 0.0000 - 0.9270*I, 0.0000 - 0.9270*I]])

        # a genus six curve
        a6 = Matrix(
            CDF,
            [
                [0.041441136367794 + 0.027833200726691*I, -0.034538213432710 + 0.027188711893010*I, -0.097853547524326 + 0.026426118256733*I, -0.304059718474855 + 0.060306772513671*I, -0.536899366726663 + 0.087203183867874*I, -2.814882797646379 + 0.543317148745872*I],
                [-0.114912806296197 - 0.044601420984340*I, 0.000000000000002 - 0.000000000000002*I, 0.053195632592273 - 0.022598899319034*I, 0.000000000000006 - 0.000000000000006*I, 0.242671354734388 + 0.009863612111086*I, 0.764112818309978 + 0.035735030643330*I],
                [0.114912806296198 + 0.016851233729331*I, -0.000000000000003 + 0.054377423786101*I, -0.053195632592272 - 0.080539669725953*I, -0.000000000000009 + 0.120613545027839*I, -0.242671354734385 - 0.237235882811205*I, -0.764112818309973 - 0.538148249906231*I],
                [-0.000000000000145 - 0.011064980468868*I, -0.000000000000232 + 0.018296506388701*I, -0.000000000000371 - 0.030253337193984*I, -0.000000000001418 + 0.111441672814486*I, -0.000000000002271 - 0.184269979844065*I, -0.000000000013900 - 1.122369328118062*I],
                [0.081988303080824 - 0.027750187255008*I, -0.000000000000001 - 0.000000000000000*I, -0.052735643727852 - 0.103138569044988*I, -0.000000000000002 - 0.000000000000001*I, -0.088120274254757 - 0.227372270700121*I, -0.127496577243115 - 0.502413219262904*I],
                [-0.081988303080827 + 0.000000000000001*I, -0.000000000000003 - 0.054377423786097*I, 0.052735643727848 + 0.0000000000000058*I, -0.000000000000006 - 0.120613545027826*I, 0.088120274254749 + 0.000000000000010*I, 0.127496577243099 + 0.000000000000019*I],
            ]
        )
        b6 = Matrix(
            CDF,
            [
                [0.041441136367794 + 0.027833200726690*I, -0.034538213432710 - 0.009404300883654*I, -0.097853547524326 + 0.026426118256743*I, -0.304059718474855 - 0.162576573110822*I, -0.536899366726663 + 0.087203183867912*I, -2.814882797646379 + 0.543317148746025*I],
                [-0.008893776107028 - 0.100911791871047*I, -0.066617940934943 + 0.018040458698884*I, 0.109123013300482 - 0.073253290304226*I, -0.116239263979475 + 0.004585936107792*I, 0.338049660213882 - 0.105519234924677*I, 0.955517317106568 - 0.261883524016635*I],
                [-0.008893776107028 - 0.056310370886708*I, -0.066617940934943 - 0.018040458698885*I, 0.109123013300482 - 0.050654390985187*I, -0.116239263979475 - 0.004585936107801*I, 0.338049660213882 - 0.115382847035744*I, 0.955517317106568 - 0.297618554659892*I],
                [0.032030533560343 - 0.000000000000020*I, 0.000000000000434 - 0.018296506388225*I, 0.142511462455675 - 0.000000000000020*I, 0.000000000002647 - 0.111441672811588*I, 0.831127378714640 - 0.000000000000147*I, 4.865652776956537 - 0.000000000001017*I],
                [0.106019030189168 + 0.028560183631701*I, 0.066617940934943 + 0.072417882484992*I, 0.055927380708208 - 0.052484178059800*I, 0.116239263979479 + 0.125199481135651*I, 0.095378305479493 - 0.111989423664385*I, 0.191404498796592 - 0.204794664603068*I],
                [-0.057957575972479 + 0.016041237352642*I, 0.066617940934944 - 0.036336965087211*I, 0.161398668163915 + 0.075083077378845*I, 0.116239263979481 - 0.116027608920027*I, 0.271618853989010 + 0.102125811553314*I, 0.446397653282827 + 0.169059633959730*I],
            ]
        )

        # store matrices
        #
        # important note! These period matrices (the ones appearing in the
        # paper) are actually transposed. all intermediate steps assume this
        # transposition
        self.atrott = atrott
        self.btrott = btrott
        self.aklein = aklein
        self.bklein = bklein
        self.afermat = afermat
        self.bfermat = bfermat
        self.a6 = a6
        self.b6 = b6

        # klein curve
        Hklein = Matrix(GF(2),[
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ])
        Qklein = Matrix(GF(2),[
            [1,1,1],
            [0,0,1],
            [0,1,0],
        ])

        # fermat curve n=4 (g=3)
        Hfermat4 = Matrix(GF(2),[
            [0,1,0],
            [1,0,0],
            [0,0,0]
        ])
        Qfermat4 = Matrix(GF(2),[
            [1,0,0],
            [0,0,1],
            [0,1,0],
        ])

        # fermat curve n=5 (g=6)
        Hfermat5 = Matrix(GF(2),[
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,1,0,0,0],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1],
        ])
        Qfermat5 = Matrix(GF(2),[
            [1,0,0,0,0,0],
            [1,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,1,0,1,0],
            [0,1,1,0,1,0],
            [0,0,1,0,1,1],
        ])

        # a genus 3 curve similar to f2
        Hf2a = Matrix(GF(2),[
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ])
        Qf2a = Matrix(GF(2),[
            [1,1,1],
            [0,0,1],
            [0,1,0],
        ])

        # a genus 6 curve
        H6 = Matrix(GF(2),[
            [0,1,0,0,0,0],
            [1,0,0,0,0,0],
            [0,0,0,1,0,0],
            [0,0,1,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
        ])
        Q6 = Matrix(GF(2),[
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [1,1,1,0,0,0],
            [0,1,0,0,1,0],
            [0,0,0,1,0,0],
            [0,0,0,0,1,1],
        ])

        # an example where it is necessary to eliminate under a 2x2 block
        N1problem = Matrix(GF(2),[
            [0,1,1,1,0,0],
            [1,0,1,0,1,1],
            [1,1,0,1,1,0],
            [1,0,1,0,0,0],
            [0,1,1,0,0,0],
            [0,1,0,0,0,0],
        ])

        # store
        self.Hklein = Hklein
        self.Qklein = Qklein
        self.N1klein = Qklein*Hklein*Qklein.T

        self.Hfermat4 = Hfermat4
        self.Qfermat4 = Qfermat4
        self.N1fermat4 = Qfermat4*Hfermat4*Qfermat4.T

        self.Hfermat5 = Hfermat5
        self.Qfermat5 = Qfermat5
        self.N1fermat5 = Qfermat5*Hfermat5*Qfermat5.T

        self.Hf2a = Hf2a
        self.Qf2a = Qf2a
        self.N1f2a = Qf2a*Hf2a*Qf2a.T

        self.H6 = H6
        self.Q6 = Q6
        self.N16 = Q6*H6*Q6.T

        self.N1problem = N1problem


class TestInvolutionMatrix(HomologyTestData):
    def test_integral(self):
        # the integrality test is built into the function itself since the
        # return type should be a matrix over ZZ
        R = involution_matrix(self.atrott, self.btrott)
        R = involution_matrix(self.aklein, self.bklein, tol=1e-3)
        R = involution_matrix(self.afermat, self.bfermat, tol=1e-3)
        R = involution_matrix(self.a6, self.b6)

    def test_eigenvalues(self):
        # the eigenvalues of the involution matrix should be equal to -1 or 1
        R = involution_matrix(self.atrott, self.btrott)
        evals = R.eigenvalues()
        self.assertSetEqual(set(evals), {1.0, -1.0})

        R = involution_matrix(self.aklein, self.bklein, tol=1e-3)
        evals = R.eigenvalues()
        self.assertSetEqual(set(evals), {1.0, -1.0})

        R = involution_matrix(self.afermat, self.bfermat, tol=1e-3)
        evals = R.eigenvalues()
        self.assertSetEqual(set(evals), {1.0, -1.0})

        R = involution_matrix(self.a6, self.b6)
        evals = R.eigenvalues()
        self.assertSetEqual(set(evals), {1.0, -1.0})

    def test_diagonalizable(self):
        # the involution matrix should be diagonalizable
        R = involution_matrix(self.atrott, self.btrott)
        RQQ = R.change_ring(QQ)
        self.assertTrue(RQQ.is_diagonalizable())

        R = involution_matrix(self.aklein, self.bklein, tol=1e-3)
        RQQ = R.change_ring(QQ)
        self.assertTrue(RQQ.is_diagonalizable())

        R = involution_matrix(self.afermat, self.bfermat, tol=1e-3)
        RQQ = R.change_ring(QQ)
        self.assertTrue(RQQ.is_diagonalizable())

        R = involution_matrix(self.a6, self.b6)
        RQQ = R.change_ring(QQ)
        self.assertTrue(RQQ.is_diagonalizable())


class TestN1Matrix(HomologyTestData):

    def test_symmetric(self):
        # the N1 matrix should be symmetric
        R = involution_matrix(self.atrott, self.btrott)
        S = integer_kernel_basis(R)
        N1 = N1_matrix(self.atrott, self.btrott, S)
        self.assertEqual(N1, N1.T)

        R = involution_matrix(self.aklein, self.bklein, tol=1e-3)
        S = integer_kernel_basis(R)
        N1 = N1_matrix(self.aklein, self.bklein, S, tol=1e-3)
        self.assertEqual(N1, N1.T)

        R = involution_matrix(self.afermat, self.bfermat, tol=1e-3)
        S = integer_kernel_basis(R)
        N1 = N1_matrix(self.afermat, self.bfermat, S, tol=1e-3)
        self.assertEqual(N1, N1.T)

        R = involution_matrix(self.a6, self.b6)
        S = integer_kernel_basis(R)
        N1 = N1_matrix(self.a6, self.b6, S)
        self.assertEqual(N1, N1.T)

class TestSymmetricBlockDiagonalization(HomologyTestData):
    def setUp(self):
        HomologyTestData.setUp(self)

        # pre-compute the N1 matrices from the period matrices. note that
        # Homology test data also contains N1 matrices "given" from the
        # Kalla,Klein paper ("given" in quotes since only the H and Q matrices
        # are actually produced)
        R = involution_matrix(self.atrott, self.btrott)
        S = integer_kernel_basis(R)
        N1 = N1_matrix(self.atrott, self.btrott, S, tol=1e-3)
        self.N1trott_from_periods = N1

        R = involution_matrix(self.aklein, self.bklein, tol=1e-3)
        S = integer_kernel_basis(R)
        N1 = N1_matrix(self.aklein, self.bklein, S, tol=1e-3)
        self.N1klein_from_periods = N1

        R = involution_matrix(self.afermat, self.bfermat, tol=1e-3)
        S = integer_kernel_basis(R)
        N1 = N1_matrix(self.afermat, self.bfermat, S, tol=1e-3)
        self.N1fermat_from_periods = N1

        R = involution_matrix(self.a6, self.b6)
        S = integer_kernel_basis(R)
        N1 = N1_matrix(self.a6, self.b6, S)
        self.N16_from_periods = N1

    def test_diagonal_locations(self):
        # tests the helper function `diagonal_locations` which identifies where
        # the diagonal elements or blocks occur in the H matrix
        H = Matrix(GF(2),
                   [[0,0,0],
                    [0,0,0],
                    [0,0,0]])
        index_one, index_B = diagonal_locations(H)
        self.assertEqual(index_one, 3)
        self.assertEqual(index_B, -1)

        H = Matrix(GF(2),
                   [[1,0,0],
                    [0,0,0],
                    [0,0,0]])
        index_one, index_B = diagonal_locations(H)
        self.assertEqual(index_one, 0)
        self.assertEqual(index_B, -1)

        H = Matrix(GF(2),
                   [[0,1,0],
                    [1,0,0],
                    [0,0,0]])
        index_one, index_B = diagonal_locations(H)
        self.assertEqual(index_one, 3)
        self.assertEqual(index_B, 0)

        H = Matrix(GF(2),
                   [[0,1,0],
                    [1,0,0],
                    [0,0,1]])
        index_one, index_B = diagonal_locations(H)
        self.assertEqual(index_one, 2)
        self.assertEqual(index_B, 0)

        H = Matrix(GF(2),
                   [[1,0,0],
                    [0,0,1],
                    [0,1,0]])
        index_one, index_B = diagonal_locations(H)
        self.assertEqual(index_one, 0)
        self.assertEqual(index_B, 1)

    def test_rank_equivalence(self):
        # the ranks of N1 and H should match
        N1 = self.N1klein
        H,Q = symmetric_block_diagonalize(N1)
        self.assertEqual(N1.rank(), H.rank())

        N1 = self.N1fermat4
        H,Q = symmetric_block_diagonalize(N1)
        self.assertEqual(N1.rank(), H.rank())

        N1 = self.N1fermat5
        H,Q = symmetric_block_diagonalize(N1)
        self.assertEqual(N1.rank(), H.rank())

        N1 = self.N1f2a
        H,Q = symmetric_block_diagonalize(N1)
        self.assertEqual(N1.rank(), H.rank())

        N1 = self.N16
        H,Q = symmetric_block_diagonalize(N1)
        self.assertEqual(N1.rank(), H.rank())

        N1 = self.N1problem
        H,Q = symmetric_block_diagonalize(N1)
        self.assertEqual(N1.rank(), H.rank())

    def test_symmetric_H(self):
        # H should be symmetric
        N1 = self.N1klein
        H,Q = symmetric_block_diagonalize(N1)
        self.assertEqual(H, H.T)

        N1 = self.N1fermat4
        H,Q = symmetric_block_diagonalize(N1)
        self.assertEqual(H, H.T)

        N1 = self.N1fermat5
        H,Q = symmetric_block_diagonalize(N1)
        self.assertEqual(H, H.T)

        N1 = self.N1f2a
        H,Q = symmetric_block_diagonalize(N1)
        self.assertEqual(H, H.T)

        N1 = self.N16
        H,Q = symmetric_block_diagonalize(N1)
        self.assertEqual(H, H.T)

        N1 = self.N1problem
        H,Q = symmetric_block_diagonalize(N1)
        self.assertEqual(H, H.T)

    def test_equivalence(self):
        # by definition: N1 = Q*H*Q.T (mod 2)
        N1 = self.N1klein
        H,Q = symmetric_block_diagonalize(N1)
        self.assertEqual(N1, Q*H*Q.T)

        N1 = self.N1fermat4
        H,Q = symmetric_block_diagonalize(N1)
        self.assertEqual(N1, Q*H*Q.T)

        N1 = self.N1fermat5
        H,Q = symmetric_block_diagonalize(N1)
        self.assertEqual(N1, Q*H*Q.T)

        N1 = self.N1f2a
        H,Q = symmetric_block_diagonalize(N1)
        self.assertEqual(N1, Q*H*Q.T)

        N1 = self.N16
        H,Q = symmetric_block_diagonalize(N1)
        self.assertEqual(N1, Q*H*Q.T)

        N1 = self.N1problem
        H,Q = symmetric_block_diagonalize(N1)
        self.assertEqual(N1, Q*H*Q.T)

    def test_integral(self):
        # symmetrize_periods already tests if Gamma is integral
        Pa, Pb = symmetrize_periods(self.atrott.T, self.btrott.T)
        Pa, Pb = symmetrize_periods(self.aklein.T, self.bklein.T, tol=1e-3)
        Pa, Pb = symmetrize_periods(self.afermat.T, self.bfermat.T, tol=1e-3)
        Pa, Pb = symmetrize_periods(self.a6.T, self.b6.T)

    def test_symplectic(self):
        # the symmetric transformation matrix should be symplectic
        Pa = self.atrott
        Pb = self.btrott
        R = involution_matrix(Pa, Pb)
        S = integer_kernel_basis(R)
        N1 = N1_matrix(Pa, Pb, S)
        H,Q = symmetric_block_diagonalize(N1)
        Gamma = symmetric_transformation_matrix(Pa, Pb, S, H, Q, tol=1e-4)
        g,g = Pa.dimensions()
        J = zero_matrix(ZZ,2*g,2*g)
        Ig = identity_matrix(ZZ, g, g)
        J[:g,g:] = Ig
        J[g:,:g] = -Ig
        self.assertEqual(Gamma.T*J*Gamma,J)

        Pa = self.aklein
        Pb = self.bklein
        R = involution_matrix(Pa, Pb, tol=1e-3)
        S = integer_kernel_basis(R)
        N1 = N1_matrix(Pa, Pb, S)
        H,Q = symmetric_block_diagonalize(N1)
        Gamma = symmetric_transformation_matrix(Pa, Pb, S, H, Q, tol=1e-3)
        g,g = Pa.dimensions()
        J = zero_matrix(ZZ,2*g,2*g)
        Ig = identity_matrix(ZZ, g, g)
        J[:g,g:] = Ig
        J[g:,:g] = -Ig
        self.assertEqual(Gamma.T*J*Gamma,J)

        Pa = self.afermat
        Pb = self.bfermat
        R = involution_matrix(Pa, Pb, tol=1e-3)
        S = integer_kernel_basis(R)
        N1 = N1_matrix(Pa, Pb, S)
        H,Q = symmetric_block_diagonalize(N1)
        Gamma = symmetric_transformation_matrix(Pa, Pb, S, H, Q, tol=1e-3)
        g,g = Pa.dimensions()
        J = zero_matrix(ZZ,2*g,2*g)
        Ig = identity_matrix(ZZ, g, g)
        J[:g,g:] = Ig
        J[g:,:g] = -Ig
        self.assertEqual(Gamma.T*J*Gamma,J)

        Pa = self.a6
        Pb = self.b6
        R = involution_matrix(Pa, Pb)
        S = integer_kernel_basis(R)
        N1 = N1_matrix(Pa, Pb, S)
        H,Q = symmetric_block_diagonalize(N1)
        Gamma = symmetric_transformation_matrix(Pa, Pb, S, H, Q, tol=1e-4)
        g,g = Pa.dimensions()
        J = zero_matrix(ZZ,2*g,2*g)
        Ig = identity_matrix(ZZ, g, g)
        J[:g,g:] = Ig
        J[g:,:g] = -Ig
        self.assertEqual(Gamma.T*J*Gamma,J)

    def test_riemann_matrix(self):
        # the transformed period matrices should still result in a Riemann
        # matrix after normalization
        Pa, Pb = symmetrize_periods(self.atrott.T, self.btrott.T)
        Omega = Pa.inverse()*Pb
        Y = Omega.apply_map(imag)
        symmetric_error = (Omega - Omega.T).norm()
        eigenvalues = Y.eigenvalues()
        self.assertLess(symmetric_error, 3e-3)  # the example itself has low precision
        for eig in eigenvalues:
            self.assertGreater(eig, 0)

        Pa, Pb = symmetrize_periods(self.aklein.T, self.bklein.T, tol=1e-3)
        Omega = Pa.inverse()*Pb
        Y = Omega.apply_map(imag)
        symmetric_error = (Omega - Omega.T).norm()
        eigenvalues = Y.eigenvalues()
        self.assertLess(symmetric_error, 1e-3)
        for eig in eigenvalues:
            self.assertGreater(eig, 0)

        Pa, Pb = symmetrize_periods(self.afermat.T, self.bfermat.T, tol=1e-3)
        Omega = Pa.inverse()*Pb
        Y = Omega.apply_map(imag)
        symmetric_error = (Omega - Omega.T).norm()
        eigenvalues = Y.eigenvalues()
        self.assertLess(symmetric_error, 1e-3)
        for eig in eigenvalues:
            self.assertGreater(eig, 0)

        Pa, Pb = symmetrize_periods(self.a6.T, self.b6.T)
        Omega = Pa.inverse()*Pb
        Y = Omega.apply_map(imag)
        symmetric_error = (Omega - Omega.T).norm()
        eigenvalues = Y.eigenvalues()
        self.assertLess(symmetric_error, 1e-3)
        for eig in eigenvalues:
            self.assertGreater(eig, 0)


    def test_recover_action(self):
        # see equation (28) of Kalla,Klein
        def compute_R(Gamma, H):
            H = H.change_ring(ZZ)
            g,g = H.dimensions()
            A = Gamma[:g,:g]
            B = Gamma[:g,g:]
            C = Gamma[g:,:g]
            D = Gamma[g:,g:]
            Ig = identity_matrix(ZZ,g)

            R = zero_matrix(ZZ,2*g,2*g)
            R[:g,:g] = (2*C.T*B - A.T*H*B + Ig).T
            R[:g,g:] = 2*D.T*B - B.T*H*B
            R[g:,:g] = -2*C.T*A + A.T*H*A
            R[g:,g:] = -(2*C.T*B - A.T*H*B + Ig)
            return R

        # trott curve
        Pa = self.atrott
        Pb = self.btrott
        R = involution_matrix(Pa, Pb)
        S = integer_kernel_basis(R)
        N1 = N1_matrix(Pa, Pb, S)
        H,Q = symmetric_block_diagonalize(N1)
        Gamma = symmetric_transformation_matrix(Pa, Pb, S, H, Q, tol=1e-4)

        Ralt = compute_R(Gamma, H)
        self.assertEqual(R, Ralt)

        # klein curve
        Pa = self.aklein
        Pb = self.bklein
        R = involution_matrix(Pa, Pb, tol=1e-3)
        S = integer_kernel_basis(R)
        N1 = N1_matrix(Pa, Pb, S, tol=1e-3)
        H,Q = symmetric_block_diagonalize(N1)
        Gamma = symmetric_transformation_matrix(Pa, Pb, S, H, Q, tol=1e-4)

        Ralt = compute_R(Gamma, H)
        self.assertEqual(R, Ralt)

        # fermat curve
        Pa = self.afermat
        Pb = self.bfermat
        R = involution_matrix(Pa, Pb, tol=1e-3)
        S = integer_kernel_basis(R)
        N1 = N1_matrix(Pa, Pb, S, tol=1e-3)
        H,Q = symmetric_block_diagonalize(N1)
        Gamma = symmetric_transformation_matrix(Pa, Pb, S, H, Q, tol=1e-4)

        Ralt = compute_R(Gamma, H)
        self.assertEqual(R, Ralt)

        # genus 6 curve
        Pa = self.a6
        Pb = self.b6
        R = involution_matrix(Pa, Pb)
        S = integer_kernel_basis(R)
        N1 = N1_matrix(Pa, Pb, S)
        H,Q = symmetric_block_diagonalize(N1)
        Gamma = symmetric_transformation_matrix(Pa, Pb, S, H, Q, tol=1e-4)

        Ralt = compute_R(Gamma, H)
        self.assertEqual(R, Ralt)


