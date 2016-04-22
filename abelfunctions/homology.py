r"""Homology

"""

from sage.all import (
    real, imag, Matrix, ZZ, QQ, RDF, CDF, GF, identity_matrix, zero_matrix)


def Re(M):
    return M.apply_map(real)

def Im(M):
    return M.apply_map(imag)


def involution_matrix(Pa, Pb, tol=1e-4):
    r"""Returns the transformation matrix `R` corresponding to the anti-holomorphic
    involution on the periods of the Riemann surface.

    Given an aritrary `2g x g` period matrix `[Pa, Pb]^T` of a genus `g`
    Riemann surface `X` the action of the anti-holomorphic involution on `X` of
    these periods is given by left-multiplication by a `2g x 2g` matrix `R`.
    That is, .. math::

        [\tau P_a^T, \tau P_b^T]^T = R [P_a^T, P_b^T]^T

    Parameters
    ----------
    Pa : complex matrix
    Pb : complex matrix
        The a- and b-periods, respectively, of a genus `g` Riemann surface.
    tol : double
        (Default: 1e-4) Tolerance used to veryify integrality of transformation
        matrix. Dependent on precision of period matrices.

    Returns
    -------
    R : complex matrix
        The anti-holomorphic involution matrix.

    Todo
    ----
    For numerical stability, replace matrix inversion with linear system
    solves.
    """
    g,g = Pa.dimensions()
    R_RDF = Matrix(RDF, 2*g, 2*g)

    Ig = identity_matrix(RDF, g)
    M = Im(Pb.T)*Re(Pa) - Im(Pa.T)*Re(Pb)
    Minv = M.inverse()

    R_RDF[:g,:g] = (2*Re(Pb)*Minv*Im(Pa.T) + Ig).T
    R_RDF[:g,g:] = -2*Re(Pa)*Minv*Im(Pa.T)
    R_RDF[g:,:g] = 2*Re(Pb)*Minv*Im(Pb.T)
    R_RDF[g:,g:] = -(2*Re(Pb)*Minv*Im(Pa.T) + Ig)
    R = R_RDF.round().change_ring(ZZ)

    # sanity check: make sure that R_RDF is close to integral. we perform this
    # test here since the matrix returned should be over ZZ
    error = (R - R_RDF).norm()
    if error > tol:
        raise ValueError("The anti-holomorphic involution matrix is not "
                         "integral. Try increasing the precision of the input "
                         "period matrices.")
    return R


def integer_kernel_basis(R):
    r"""Returns the Z-basis `[S1 \\ S2]` of the kernel of the anti-holomorphic
    involution matrix `R`.

    The `2g x g` matrix `[S1 \\ S2]` represents a Z-basis of the kernel space
    .. math::

        K_\mathbb{Z} = \text{ker}(R^T - \mathbb{I}_{2g})

    That is, the basis of the space of all vectors fixed by the
    anti-holomorphic involution `R`.

    Used as input in `N1_matrix`.

    Parameters
    ----------
    R : integer matrix
        The anti-holomorphic involution matrix of a genus `g` Riemann surface.

    Returns
    -------
    S : integer matrix
        A `2g x g` matrix where each column is a basis element of the fixed
        point space of `R`.

    """
    twog, twog = R.dimensions()
    g = twog/2
    K = R.T - identity_matrix(ZZ, twog)
    r = K.rank()

    # sanity check: the rank of the kernel should be the genus of the curve
    if r != g:
        raise ValueError("The rank of the integer kernel of K should be "
                         "equal to the genus.")

    # compute the integer kernel from the Smith normal form of K
    D,U,V = K.smith_form()
    S = V[:,g:]
    return S

def N1_matrix(Pa, Pb, S, tol=1e-4):
    r"""Returns the matrix `N1` from the integer kernel of the anti-holomorphic
    involution matrix.

    This matrix `N1` is used directly to determine the topological type of a
    Riemann surface. Used as input in `symmetric_block_diagonalize`.

    Paramters
    ---------
    S : integer matrix
        A `2g x g` Z-basis of the kernel of the anti-holomorphic involution.
        (See `integer_kernel_basis`.)
    tol : double
        (Default: 1e-4) Tolerance used to veryify integrality of the matrix.
        Dependent on precision of period matrices.

    Returns
    -------
    N1 : GF(2) matrix
        A `g x g` matrix from which we can compute the topological type.

    """
    # compute the Smith normal form of S, itself
    g = S.ncols()
    S1 = S[:g,:]
    S2 = S[g:,:]
    ES, US, VS = S.smith_form()

    # construct the matrix N1 piece by piece
    Nper = zero_matrix(RDF, 2*g,g)
    Nper[:g,:] = -Re(Pb)[:,:]
    Nper[g:,:] = Re(Pa)[:,:]
    Nhat = (S1.T*Re(Pa) + S2.T*Re(Pb)).inverse()
    Ntilde = 2*US*Nper*Nhat
    N1_RDF = VS*Ntilde[:g,:]
    N1 = N1_RDF.round().change_ring(GF(2))

    # sanity check: N1 should be integral
    error = (N1_RDF.round() - N1_RDF).norm()
    if error > tol:
        raise ValueError("The N1 matrix is not integral. Try increasing the "
                         "precision of the input period matrices.")
    return N1

