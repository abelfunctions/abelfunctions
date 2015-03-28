/*=============================================================================

  riemanntheta.h

  Computing the Riemann theta function.


  Authors:
  -------

  * Chris Swierczewski (cswiercz@uw.edu) - September 2012
  * Grady Williams (gradyrw@uw.edu) - November 2012

=============================================================================*/


#ifndef __RIEMANNTHETA_H__
#define __RIEMANNTHETA_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */




/******************************************************************************
  finite_sum_without_derivatives
  ------------------------------

  Computes the real and imaginary parts of the finite sum.

  Output:

  * fsum_real, fsum_imag: the real and imaginary parts of the finite sum

  Input:

  * X, Yinv, T: row-major matrices such that the Riemann matrix, Omega is 
                equal to (X + iY). T is the Cholesky decomposition of Y

  * x, y: the real and imaginary parts of the input vector, z.
  * S: the set of points in ZZ^g over which to compute the finite sum
  * g: the dimension of the above matrices and vectors
  * N: the number of points in ZZ^g over which to compute the sum
       (= total number of elements in S / g)
******************************************************************************/
void
finite_sum_without_derivatives(double* fsum_real, double* fsum_imag,
			       double* X, double* Yinv, double* T,
			       double* x, double* y, double* S,
			       int g, int N);



/******************************************************************************
  finite_sum_with_derivatives
  ------------------------------

  Computes the real and imaginary parts of the finite sum.

  Output:

  * fsum_real, fsum_imag: the real and imaginary parts of the finite sum

  Input:

  * X, Yinv, T: row-major matrices such that the Riemann matrix, Omega is 
                equal to (X + iY). T is the Cholesky decomposition of Y

  * x, y: the real and imaginary parts of the input vector, z.
  * S: the set of points in ZZ^g over which to compute the finite sum
  * deriv_real, deriv_imag: the real and imaginary parts of the derivative
    vectors
  * nderivs: the number of derivative vectors. Thus, deriv_real and deriv_imag
    have a total of g*nderivs elements.
  * g: the dimension of the above matrices and vectors
  * N: the number of points in ZZ^g over which to compute the sum
       (= total number of elements in S / g)
******************************************************************************/
void
finite_sum_with_derivatives(double* fsum_real, double* fsum_imag,
			    double* X, double* Yinv, double* T,
			    double* x, double* y, double* S,
			    double* deriv_real, double* deriv_imag, int nderivs,
			    int g, int N);




#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif /* __RIEMANNTHETA_H__ */
