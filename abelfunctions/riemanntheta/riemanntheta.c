/*=============================================================================

  riemanntheta.c

  Computing the Riemann theta function.


  Authors:
  -------

  *Chris Swierczewski (cswiercz@uw.edu) - September 2012
  *Grady Williams (gradyrw@uw.edu) - October 2012

=============================================================================*/

#ifndef __RIEMANNTHETA_H__
#define __RIEMANNTHETA_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/******************************************************************************
  exppart
  -------

  A helper function for the finite sum functions. Computes

	     2pi < (n-intshift), (1/2)X(n-intshift) + x >

******************************************************************************/

double
exppart(double* n, double* X, double* x, double* intshift, int g)
{
    double tmp1[g], tmp2[g];
    int i,j;

    //tmp1 = n - intshift
    for (i = 0; i < g; i++) {
         tmp1[i] = n[i] - intshift[i];
    }	

    // tmp2 = (1/2)X*(n-intshift)
    double sum;
    for (i = 0; i < g; i++) {
    	sum = 0;
    	for (j = 0; j < g; j++) {
    		sum += X[i*g + j] * tmp1[j];
    	}
    	tmp2[i] = sum/2;
    }
	
    //tmp2 = (1/2)*X(n-intshift) + x
    for (i = 0; i < g; i++){
    	tmp2[i] = tmp2[i] + x[i];
    }
    //ept = <tmp1,tmp2>
    double dot = 0;
    for (i = 0; i < g; i++){
    	dot += tmp1[i]*tmp2[i];
    }
    double ept = 2* M_PI * dot; 
    return ept;
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

    double tmp1[g], tmp2[g];
    int i,j;

    //tmp1 = n + fracshift
    for (i = 0; i < g; i++) {
    	tmp1[i] = n[i] + fracshift[i];
    }
	
    //tmp2 = T*(n+fracshift)
    double sum;
    for (i = 0; i < g; i++) {
    	sum = 0;
    	for (j = 0; j < g; j++) {
    		sum += T[i*g + j] * tmp1[j];
    	}
    	tmp2[i] = sum;
    }
 
    //norm = || T*(n + fracshift) || ^ 2
    double norm = 0;
    for (i = 0; i < g; i++) {
    	norm += tmp2[i] * tmp2[i];
    }
    return -M_PI * norm;
}


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
	                       int g, int N)
{
    /* 
     compute the shifted vectors: shift = Yinv*y and its 
     integer and fractional parts 
    */

    int k,j;
    //double shift[g], intshift[g], fracshift[g];

    double *intshift;
    double *shift;
    double *fracshift;
    shift = (double*)malloc(g*sizeof(double));
    intshift = (double*)malloc(g*sizeof(double));
    fracshift = (double*)malloc(g*sizeof(double));

    // shift = Yinv*y;
    // intshift = round(shift)
    // fracshift = shift - intshift
    double sum;
    for (k = 0; k < g; k++) {
        sum = 0;
       	for (j = 0; j < g; j++) {
	    sum += Yinv[k*g + j] * y[j];
	}
	shift[k] = sum;
    }

    for(k = 0; k < g; k++) {
        intshift[k] = round(shift[k]);
        fracshift[k] = shift[k] - intshift[k];
    }

    //compute the finite sum
    double real_total = 0, imag_total = 0;
    double ept, npt, cpt, spt;
    double *n;
    for(k=0; k<N; k++) {
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
    *fsum_real = real_total;
    *fsum_imag = imag_total;

    //free any allocated vectors
    free(shift);
    free(intshift);
    free(fracshift);
}

/******************************************************************************
  deriv_prod
  ----------

  compute the real and imaginary parts of the product
                   ___
                   | |    2*pi*I <d, n-intshift>
	           | |
	       d in derivs

  for a given n in ZZ^g.
******************************************************************************/
void
deriv_prod(double* dp_real, double* dp_imag,
           double* n, double* intshift,
           double* deriv_real, double* deriv_imag, int nderivs,
           int g)
{

    double nmintshift[g];
    double term_real = 0;
    double term_imag = 0;
    int i,j;

    // compute n-intshift
    for (i = 0; i < g; i++) {
        nmintshift[i] = n[i] - intshift[i];
    }
    
    for (i = 0; i < g; i++) {
      term_real += deriv_real[i] * nmintshift[i];
      term_imag += deriv_imag[i] * nmintshift[i];
    }
    dp_imag[0] = 2*M_PI*term_real;
    dp_real[0] = -2*M_PI*term_imag;
}


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
		            double* deriv_real, double* deriv_imag, 
                            int nderivs, int g, int N)
{
  /* 
     compute the shifted vectors: shift = Yinv*y and its 
     integer and fractional parts 
    */

    int k,j;
    //double shift[g], intshift[g], fracshift[g];

    double *intshift;
    double *shift;
    double *fracshift;
    shift = (double*)malloc(g*sizeof(double));
    intshift = (double*)malloc(g*sizeof(double));
    fracshift = (double*)malloc(g*sizeof(double));

    // shift = Yinv*y;
    // intshift = round(shift)
    // fracshift = shift - intshift
    double sum;
    for (k = 0; k < g; k++) {
        sum = 0;
       	for (j = 0; j < g; j++) {
	    sum += Yinv[k*g + j] * y[j];
	}
	shift[k] = sum;
    }

    for(k = 0; k < g; k++) {
        intshift[k] = round(shift[k]);
        fracshift[k] = shift[k] - intshift[k];
    }

    //compute the finite sum
    double real_total = 0, imag_total = 0;
    double ept, npt, cpt, spt;
    double dpr[0];
    double dpi[0];
    dpr[0] = 0;
    dpi[0] = 0;
    double *n;
    for(k=0; k < N; k++) {
        // the current point in S \subset ZZ^g
        n = S + k*g;

        // compute the "cosine" and "sine" parts of the summand
        ept = exppart(n, X, x, intshift, g);
        npt = exp(normpart(n, T, fracshift, g));
        cpt = npt * cos(ept);
        spt = npt * sin(ept);
	deriv_prod(dpr,dpi, n, intshift, deriv_real, deriv_imag,nderivs, g);
        real_total += dpr[0] * cpt - dpi[0] * spt;
        imag_total += dpi[0] * cpt + dpr[0] * spt;
    }
 
    //store values to poiners
    *fsum_real = real_total;
    *fsum_imag = imag_total;

    //free any allocated vectors
    free(shift);
    free(intshift);
    free(fracshift);     

}

/******************************************************************************
  integer_points

  Compute the set U_R of the integeral points needed to compute the Riemann 
  theta function to the given precision.
******************************************************************************/
/*
int
integer_points(double* intpoints, 
	       double* Yinv, 
	       double* T, 
	       double* Tinv, 
	       double* x, 
	       double* y, 
	       int g, 
	       double R, 
	       bool use_uniform)
{
  // determine center of ellipsoid
  double* c     = (double*) malloc(g*sizeof(double));
  double* intc  = (double*) malloc(g*sizeof(double));
  double* leftc = (double*) malloc(g*sizeof(double));
  int i,j,k;


  if (use_uniform)
    {
      for (i=0; i<g; i++)
	{
	  c[i]     = 0; 
	  intc[i]  = 0; 
	  leftc[i] = 0;
	}
    }
  else
    {
      // c = Yinv * y; intc = round(c); leftc = c - intc
      cblas_dgemv(CblasRowMajor, CblasNoTrans,
		  g, g, 1.0, Yinv,
		  g, y, 1, 0, c, 1);
      for (i=0; i<g; i++)
	{
	  intc[i]  = round(c[i]); 
	  leftc[i] = c[i] - intc[i];
	}
    }
  
  // recursively call find_integer_points
  find_integer_points(intpoints, T, start, c, g, R);

  // free allocated memory
  free(c);
  free(intc);
  free(leftc);
}


int
find_integer_points(double* intpoints, 
		    double* T, 
		    double* start,
		    double* c,
		    int g,
		    int gcurr,
		    double R,
		    int total_intpoints)
{
  int a, b;
  a = (int) ceil(c[gcurr]-R/T[gcurr*g+gcurr-1]);
  b = (int) ceil(c[gcurr]+R/T[gcurr*g+gcurr-1]);

  // check if we reached the edge of the ellipsoid
  if (a >= b)
    return 0;

  // last dimension reached: append points to every
  // element of start
  //
  // In Python:
  //
  //     [np.append([i],start) for i in range(a,b+1)]
  //
  // so add the vectors of the form [a,start[0],...,start[gcurr-1]], ..., 
  // [b,start[0],...,start[gcurr-1]].
  //
  if (gcurr == 0)
    {
      total_intpoints += b-a;
      intpoints = (double*) realloc(intpoints, total_intpoints*sizeof(double));
    }


  // compute new shifts, radii, start, and recurse
  //double* newT = malloc(
  int newg;
  newg = g-1;

  free(newT);
  free(newTinv);
  
}
*/

#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif /* __RIEMANNTHETA_H__ */
