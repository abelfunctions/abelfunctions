r"""Homology

"""

from sage.all import (
    real, imag, Matrix, ZZ, QQ, RDF, CDF, GF, identity_matrix, round)


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
    R = R_RDF.apply_map(round).change_ring(ZZ)

    # sanity check: make sure that R_RDF is close to integral. we perform this
    # test here since the matrix returned should be over ZZ
    error = (R - R_RDF).norm()
    if error > tol:
        raise ValueError("The anti-holomorphic involution matrix is not "
                         "integral. Try increasing the precision of the input "
                         "period matrices.")
    return R

