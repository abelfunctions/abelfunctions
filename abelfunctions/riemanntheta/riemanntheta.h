
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

