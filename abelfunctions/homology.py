r"""
Homology :mod:`homology`
========================

Tools for "symmetrizing" a period matrix.

There exists a symplectic transformation on the period matrix of a real curve
such that the corresponding a- and b-cycles have certain transformation
properties until the anti-holomorphic involution on said Riemann surface.

.. note::

   The algorithm described in Kalla, Klein actually operates on the transposes
   of the a- and b-period matrices. All intermediate functions assume the input
   period matrices are transposed. The primary function in this module,
   :func:`symmetrize_periods`

Functions
---------

.. autosummary::

    symmetrize_periods
    symmetric_transformation_matrix

References
----------

.. [KallaKlein] C. Kalla, C. Klein "Computation of the Topological Type of a
   Real Riemann Surface"

Contents
--------

"""

import numpy
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
    error = (R_RDF.round() - R_RDF).norm()
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
    g = twog//2
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


def symmetric_block_diagonalize(N1):
    r"""Returns matrices `H` and `Q` such that `N1 = Q*H*Q.T` and `H` is block
    diagonal.

    The algorithm used here is as follows. Whenever a row operation is
    performed (via multiplication on the left by a transformation matrix `q`)
    the corresponding symmetric column operation is also performed via
    multiplication on the right by `q^T`.

    For each column `j` of `N1`:

    1. If column `j` consists only of zeros then swap with the last column with
       non-zero entries.

    2. If there is a `1` in position `j` of the column (i.e. a `1` lies on the
       diagonal in this column) then eliminate further entries below as in
       standard Gaussian elimination.

    3. Otherwise, if there is a `1` in the column, but not in position `j` then
       rows are swapped in a way that it appears in the position `j+1` of the
       column. Eliminate further entries below as in standard Gaussian
       elimination.

    4. After elimination, if `1` lies on the diagonal in column `j` then
       increment `j` by one. If instead the block matrix `[0 1 \\ 1 0]` lies
       along the diagonal then eliminate under the `(j,j+1)` element (the upper
       right element) of this `2 x 2` block and increment `j` by two.

    5. Repeat until `j` passes the final column or until further columns
       consists of all zeros.

    6. Finally, perform the appropriate transformations such that all `2 x 2`
       blocks in `H` appear first in the diagonalization. (Uses the
       `diagonal_locations` helper function.)

    Parameters
    ----------
    N1 : GF(2) matrix

    Returns
    -------
    H : GF(2) matrix
        Symmetric `g x g` matrix where the diagonal elements consist of either
        a "1" or a `2 x 2` block matrix `[0 1 \\ 1 0]`.
    Q : GF(2) matrix
        The corresponding transformation matrix.
    """
    g = N1.nrows()
    H = zero_matrix(GF(2), g)
    Q = identity_matrix(GF(2), g)

    # if N1 is the zero matrix the H is also the zero matrix (and Q is the
    # identity transformation)
    if (N1 % 2) == 0:
        return H,Q

    # perform the "modified gaussian elimination"
    B = Matrix(GF(2),[[0,1],[1,0]])
    H = N1.change_ring(GF(2))
    j = 0
    while (j < g) and (H[:,j:] != 0):
        # if the current column is zero then swap with the last non-zero column
        if H.column(j) == 0:
            last_non_zero_col = max(k for k in range(j,g) if H.column(k) != 0)
            Q.swap_columns(j,last_non_zero_col)
            H = Q.T*N1*Q

        # if the current diagonal element is 1 then gaussian eliminate as
        # usual. otherwise, swap rows so that a "1" appears in H[j+1,j] and
        # then eliminate from H[j+1,j]
        if H[j,j] == 1:
            rows_to_eliminate = (r for r in range(g) if H[r,j] == 1 and r != j)
            for r in rows_to_eliminate:
                Q.add_multiple_of_column(r,j,1)
            H = Q.T*N1*Q
        else:
            # find the first non-zero element in the column after the diagonal
            # element and swap rows with this element
            first_non_zero = min(k for k in range(j,g) if H[k,j] != 0)
            Q.swap_columns(j+1,first_non_zero)
            H = Q.T*N1*Q

            # eliminate *all* other ones in the column, including those above
            # the element (j,j+1)
            rows_to_eliminate = (r for r in range(g) if H[r,j] == 1 and r != j+1)
            for r in rows_to_eliminate:
                Q.add_multiple_of_column(r,j+1,1)
            H = Q.T*N1*Q

        # increment the column based on the diagonal element
        if H[j,j] == 1:
            j += 1
        elif H[j:(j+2),j:(j+2)] == B:
            # in the block diagonal case, need to eliminate below the j+1 term
            rows_to_eliminate = (r for r in range(g) if H[r,j+1] == 1 and r != j)
            for r in rows_to_eliminate:
                Q.add_multiple_of_column(r,j,1)
            H = Q.T*N1*Q
            j += 2

    # finally, check if there are blocks of "special" form. that is, shift all
    # blocks such that they occur first along the diagonal of H
    index_one, index_B = diagonal_locations(H)
    while index_one < index_B:
        j = index_B

        Qtilde = zero_matrix(GF(2), g)
        Qtilde[0,0] = 1
        Qtilde[j,0] = 1; Qtilde[j+1,0] = 1
        Qtilde[0,j] = 1; Qtilde[0,j+1] = 1
        Qtilde[j:(j+2),j:(j+2)] = B

        Q = Q*Qtilde
        H = Q.T*N1*Q

        # continue until none are left
        index_one, index_B = diagonal_locations(H)

    # above, we used Q to store column operations on N1. switch to rows
    # operations on H so that N1 = Q*H*Q.T
    Q = Q.T.inverse()
    return H,Q

def diagonal_locations(H):
    r"""Returns the indices of the last `1` along the diagonal and the first block
    along the diagonal of `H`.

    Parameters
    ----------
    H : symmetric GF(2) matrix
        Contains either 1's along the diagonal or anti-symmetric blocks.

    Returns
    -------
    index_one : integer
        The last occurrence of a `1` along the diagonal of `H`. Equal to `g`
        if there are no ones along the diagonal.
    index_B : integer
        The first occurrence of a block along the diagonal of `H`. Equal to
        `-1` if there are no blocks along the diagonal.

    """
    g = H.nrows()
    B = Matrix(GF(2),[[0,1],[1,0]])
    try:
        index_one = min(j for j in range(g) if H[j,j] == 1)
    except ValueError:
        index_one = g

    try:
        index_B = max(j for j in range(g-1) if H[j:(j+2),j:(j+2)] == B)
    except ValueError:
        index_B = -1

    return index_one, index_B


def symmetric_transformation_matrix(Pa, Pb, S, H, Q, tol=1e-4):
    r"""Returns the symplectic matrix `\Gamma` mapping the period matrices `Pa,Pb`
    to a symmetric period matrices.

    A helper function to :func:`symmetrize_periods`.

    Parameters
    ----------
    Pa : complex matrix
        A `g x g` a-period matrix.
    Pb : complex matrix
        A `g x g` b-period matrix.
    S : integer matrix
        Integer kernel basis matrix.
    H : integer matrix
        Topological type classification matrix.
    Q : integer matrix
        The transformation matrix from `symmetric_block_diagonalize`.
    tol : double
        (Default: 1e-4) Tolerance used to verify integrality of intermediate
        matrices. Dependent on precision of period matrices.

    Returns
    -------
    Gamma : integer matrix
        A `2g x 2g` symplectic matrix.
    """
    # compute A and B
    g,g = Pa.dimensions()
    rhs = S*Q.change_ring(ZZ)
    A = rhs[:g,:g].T
    B = rhs[g:,:g].T
    H = H.change_ring(ZZ)

    # compute C and D
    half = QQ(1)/QQ(2)
    temp = (A*Re(Pa) + B*Re(Pb)).inverse()
    CT = half*A.T*H - Re(Pb)*temp
    CT_ZZ = CT.round().change_ring(ZZ)
    C = CT_ZZ.T

    DT = half*B.T*H + Re(Pa)*temp
    DT_ZZ = DT.round().change_ring(ZZ)
    D = DT_ZZ.T

    # sanity checks: make sure C and D are integral
    C_error = (CT.round() - CT).norm()
    D_error = (DT.round() - DT).norm()
    if (C_error > tol) or (D_error > tol):
        raise ValueError("The symmetric transformation matrix is not integral. "
                         "Try increasing the precision of the input period "
                         "matrices.")

    # construct Gamma
    Gamma = zero_matrix(ZZ, 2*g, 2*g)
    Gamma[:g,:g] = A
    Gamma[:g,g:] = B
    Gamma[g:,:g] = C
    Gamma[g:,g:] = D
    return Gamma

def symmetrize_periods(Pa, Pb, tol=1e-4):
    r"""Returns symmetric a- and b-periods `Pa_symm` and `Pb_symm`, as well as the
    corresponding symplectic operator `Gamma` such that `Gamma [Pa \\ Pb] =
    [Pa_symm \\ Pb_symm]`.

    Parameters
    ----------
    Pa : complex matrix
    Pb : complex matrix
        The a- and b-periods, respectively, of a genus `g` Riemann surface.
    tol : double
        (Default: 1e-4) Tolerance used to verify integrality of intermediate
        matrices. Dependent on precision of period matrices.

    Returns
    -------
    Gamma : integer matrix
        The symplectic transformation operator.
    Pa : complex matrix
    Pb : complex matrix
        Symmetric a- and b-periods, respectively, of a genus `g` Riemann surface.

    Notes
    -----
    The algorithm described in Kalla, Klein actually operates on the transposes
    of the a- and b-period matrices.
    """
    # coerce from numpy, if necessary
    if isinstance(Pa, numpy.ndarray):
        Pa = Matrix(CDF, numpy.ascontiguousarray(Pa))
    if isinstance(Pb, numpy.ndarray):
        Pb = Matrix(CDF, numpy.ascontiguousarray(Pb))

    # use the transposes of the period matrices and coerce to Sage matrices
    Pa = Pa.T
    Pb = Pb.T

    # use above functions to obtain topological type matrix
    g,g = Pa.dimensions()
    R = involution_matrix(Pa, Pb, tol=tol)
    S = integer_kernel_basis(R)
    N1 = N1_matrix(Pa, Pb, S, tol=tol)
    H,Q = symmetric_block_diagonalize(N1)
    Gamma = symmetric_transformation_matrix(Pa, Pb, S, H, Q, tol=tol)

    # compute the corresponding symmetric periods
    stacked_periods = zero_matrix(CDF, 2*g, g)
    stacked_periods[:g,:] = Pa
    stacked_periods[g:,:] = Pb
    stacked_symmetric_periods = Gamma*stacked_periods
    Pa_symm = stacked_symmetric_periods[:g,:]
    Pb_symm = stacked_symmetric_periods[g:,:]

    # transpose results back
    Pa_symm = Pa_symm.T
    Pb_symm = Pb_symm.T
    return Pa_symm, Pb_symm
