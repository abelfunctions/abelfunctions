/*=============================================================================

  finite_sum.c

  Efficiently computing the finite sum part of the Riemann theta function.

  Functions
  ---------
  exppart
  normpart
  finite_sum_without_derivatives
  deriv_prod
  finite_sum_with_derivatives

  Authors
  -------

  * Chris Swierczewski (@cswiercz) - September 2012, July 2016
  * Grady Williams (@gradyrw) - October 2012
  * Jeremy Upsal (@jupsal) - July 2016

=============================================================================*/

#ifndef __FINITE_SUM_C__
#define __FINITE_SUM_C__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/******************************************************************************
  exppart
  -------

  A helper function for the finite sum functions. Computes

	     2pi < (n-intshift), (1/2)X(n-intshift) + x >

******************************************************************************/

double
exppart(double* n, double* X, double* x, double* intshift, int g)
{
  double tmp1[g];
  double tmp2[g];
  int i,j;

  // tmp1 = n - intshift
  for (i = 0; i < g; i++)
    tmp1[i] = n[i] - intshift[i];

  // tmp2 = (1/2)X*(n-intshift)
  double sum;
  for (i = 0; i < g; i++) {
    sum = 0;
    for (j = 0; j < g; j++)
      sum += X[i*g + j] * tmp1[j];

    tmp2[i] = sum/2.0;
  }

  // tmp2 = (1/2)*X(n-intshift) + x
  for (i = 0; i < g; i++)
    tmp2[i] = tmp2[i] + x[i];

  // ept = <tmp1,tmp2>
  double dot = 0;
  for (i = 0; i < g; i++)
    dot += tmp1[i]*tmp2[i];

  return 2* M_PI * dot;
}


/******************************************************************************
  normpart
  --------

  A helper function for the finite sum functions. Computes

	     -pi * || T*(n+fracshift) ||^2

******************************************************************************/
double
normpart(double* n, double* T, double* fracshift, int g)
{
  double tmp1[g];
  double tmp2[g];
  int i,j;

  // tmp1 = n + fracshift
  for (i = 0; i < g; i++)
    tmp1[i] = n[i] + fracshift[i];

  // tmp2 = T*(n+fracshift)
  double sum;
  for (i = 0; i < g; i++) {
    sum = 0;
    for (j = 0; j < g; j++)
      sum += T[i*g + j] * tmp1[j];

    tmp2[i] = sum;
  }

  // norm = || T*(n + fracshift) || ^ 2
  double norm = 0;
  for (i = 0; i < g; i++)
    norm += tmp2[i] * tmp2[i];

  return -M_PI * norm;
}


/******************************************************************************
  finite_sum_without_derivatives
  ------------------------------

  Computes the real and imaginary parts of the finite sum.

  Parameters
  ----------
  X, Yinv, T : double[:]
      Row-major matrices such that the Riemann matrix, Omega is equal to (X +
      iY). T is the Cholesky decomposition of Y.
  x, y : double[:]
      The real and imaginary parts of the input vector, z.
  S : double[:]
      The set of points in ZZ^g over which to compute the finite sum
  g : int
      The dimension of the above matrices and vectors
  N : int
     The number of points in ZZ^g over which to compute the sum
     (= total number of elements in S / g)

  Returns
  -------
  * fsum_real, fsum_imag : double*
        The real and imaginary parts of the finite sum.

******************************************************************************/

void
finite_sum_without_derivatives(double* fsum_real, double* fsum_imag,
                               double* X, double* Yinv, double* T,
                               double* x, double* y, double* S,
                               int g, int N)
{
  // compute the shifted vectors: shift = Yinv*y as well as its integer and
  // fractional parts
  int k,j;
  double shift[g];
  double intshift[g];
  double fracshift[g];

  // compute the following:
  //   * shift = Yinv*y;
  //   * intshift = round(shift)  ( or should it be floor?!?)
  //   * fracshift = shift - intshift
  double sum;
  for (k = 0; k < g; k++) {
    sum = 0;
    for (j = 0; j < g; j++)
      sum += Yinv[k*g + j] * y[j];

    shift[k] = sum;
  }

  for(k = 0; k < g; k++) {
    intshift[k] = round(shift[k]);
    fracshift[k] = shift[k] - intshift[k];
  }

  // compute the finite sum
  double real_total = 0, imag_total = 0;
  double ept, npt, cpt, spt;
  double* n;
  for(k = 0; k < N; k++) {
    // the current point in S \subset ZZ^g
    n = S + k*g;

    // compute the "cosine" and "sine" parts of the summand
    ept = exppart(n, X, x, intshift, g);
    npt = exp(normpart(n, T, fracshift, g));
    cpt = npt * cos(ept);
    spt = npt * sin(ept);
    real_total += cpt;
    imag_total += spt;
  }

  //store values to poiners
  fsum_real[0] = real_total;
  fsum_imag[0] = imag_total;
}

/******************************************************************************
  deriv_prod
  ----------

  Compute the real and imaginary parts of the product
             ___
             | |    2*pi*I <d, n-intshift>
	           | |
	       d in derivs

  for a given n in ZZ^g.

  Parameters
  ----------
  n : double[:]
      An integer vector in the finite sum ellipsoid.
  intshift : double[:]
      The integer part of Yinv*y.
  deriv_real, deriv_imag : double[:]
      The real and imaginary parts of the derivative directional vectors.
  nderivs : int
      Number / order of derivatives.
  g : int
      Genus / dimension of the problem.

  Returns
  -------
  * dpr, dpi : double
        The real and imaginary parts of the "derivative product".
******************************************************************************/
void
deriv_prod(double* dpr, double* dpi,
           double* n, double* intshift,
           double* deriv_real, double* deriv_imag, int nderivs,
           int g)
{
  double nmintshift[g];
  double term_real, term_imag;
  double total_real, total_real_tmp;
  double total_imag, total_imag_tmp;
  int i,j;

  // compute n-intshift
  for (i = 0; i < g; i++)
    nmintshift[i] = n[i] - intshift[i];

  /*
     Computes the dot product of each directional derivative and nmintshift.
     Then it computes the product of the resulting complex scalars.
  */
  total_real = 1;
  total_imag = 0;
  for (i = 0; i < nderivs; i++) {
    term_real = 0;
    term_imag = 0;
    for (j = 0; j < g; j++) {
      term_real += deriv_real[j + g*i] * nmintshift[j];
      term_imag += deriv_imag[j + g*i] * nmintshift[j];
    }

    /*
      Multiplies the dot product that was just computed with the product of all
      the previous terms. Total_real is the resulting real part of the sum, and
      total_imag is the resulting imaginary part.
    */
    total_real_tmp = total_real * term_real - total_imag * term_imag;
    total_imag_tmp = total_real * term_imag + total_imag * term_real;
    total_real = total_real_tmp;
    total_imag = total_imag_tmp;
  }

  // Compute (2*pi*i)^(nderivs) * (total_real + total_imag*i)
  double pi_mult = pow(2*M_PI, (double)nderivs);

  /*
    Determines what the result of i^nderivs is, and performs the correct
    multiplication afterwards.
  */
  if (nderivs % 4 == 0) {
    dpr[0] = pi_mult * total_real;
    dpi[0] = pi_mult * total_imag;
  }
  else if (nderivs % 4 == 1) {
    dpr[0] = -pi_mult * total_imag;
    dpi[0] = pi_mult * total_real;
  }
  else if (nderivs % 4 == 2) {
    dpr[0] = -pi_mult * total_real;
    dpi[0] = -pi_mult * total_imag;
  }
  else if (nderivs % 4 == 3) {
    dpr[0] = pi_mult * total_imag;
    dpi[0] = -pi_mult * total_real;
  }
}


/******************************************************************************
  finite_sum_with_derivatives
  ------------------------------

  Computes the real and imaginary parts of the finite sum with derivatives.

  Parameters
  ----------
  X, Yinv, T : double[:]
      Row-major matrices such that the Riemann matrix, Omega is equal to (X +
      iY). T is the Cholesky decomposition of Y.
  x, y : double[:]
      The real and imaginary parts of the input vector, z.
  S : double[:]
      The set of points in ZZ^g over which to compute the finite sum
  deriv_real, deriv_imag : double[:]
      The real and imaginary parts of the derivative directional vectors.
  nderivs : int
      Number / order of derivatives.
  g : int
      The dimension of the above matrices and vectors
  N : int
     The number of points in ZZ^g over which to compute the sum
     (= total number of elements in S / g)

  Returns
  -------
  fsum_real, fsum_imag : double*
      The real and imaginary parts of the finite sum.

******************************************************************************/
void
finite_sum_with_derivatives(double* fsum_real, double* fsum_imag,
                            double* X, double* Yinv, double* T,
                            double* x, double* y, double* S,
                            double* deriv_real, double* deriv_imag,
                            int nderivs, int g, int N)
{
  /*
    compute the shifted vectors: shift = Yinv*y as well as its integer and
    fractional parts
  */
  int k,j;
  double shift[g];
  double intshift[g];
  double fracshift[g];
  double sum;
  for (k = 0; k < g; k++) {
    sum = 0;
    for (j = 0; j < g; j++)
      sum += Yinv[k*g + j] * y[j];

    shift[k] = sum;
  }

  for(k = 0; k < g; k++) {
    intshift[k] = round(shift[k]);
    fracshift[k] = shift[k] - intshift[k];
  }

  // compute the finite sum
  double real_total = 0, imag_total = 0;
  double ept, npt, cpt, spt;
  double dpr[1];
  double dpi[1];
  double* n;
  dpr[0] = 0;
  dpi[0] = 0;
  for(k = 0; k < N; k++) {
    // the current point in S \subset ZZ^g
    n = S + k*g;

    // compute the "cosine" and "sine" parts of the summand
    ept = exppart(n, X, x, intshift, g);
    npt = exp(normpart(n, T, fracshift, g));
    cpt = npt * cos(ept);
    spt = npt * sin(ept);
    deriv_prod(dpr, dpi, n, intshift, deriv_real, deriv_imag, nderivs, g);
    real_total += dpr[0] * cpt - dpi[0] * spt;
    imag_total += dpr[0] * spt + dpi[0] * cpt;
  }

  // store values to poiners
  fsum_real[0] = real_total;
  fsum_imag[0] = imag_total;
}

#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif /* __FINITE_SUM_C__ */
