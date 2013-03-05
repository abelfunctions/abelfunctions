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
deriv_prod(double* dpr, double* dpi,
           double* n, double* intshift,
           double* deriv_real, double* deriv_imag, int nderivs,
           int g)
{

    double nmintshift[g];
    double term_real, term_imag;

    //Note that total_real + total_imag*i = 1 + 0i = 1
    double total_real = 1;
    double total_imag = 0;
    int i,j;

    // compute n-intshift
    for (i = 0; i < g; i++) {
        nmintshift[i] = n[i] - intshift[i];
    }
    
    /*Computes the dot product of each directional derivative and nmintshift
      Then it computes the product of the resulting complex scalars*/
    for (i = 0; i < nderivs; i++) {
        term_real = 0;
        term_imag = 0;
        for (j = 0; j < g; j++) {
            term_real += deriv_real[j + g*i] * nmintshift[j];
            term_imag += deriv_imag[j + g*i] * nmintshift[j];
        }

	/*Multiplies the dot product that was just computed with the product of
	all the previous terms. Total_real is the resulting real part of the 
	sum, and total_imag is the resulting imaginary part.*/
        total_real  = total_real * term_real - total_imag * term_imag;
	total_imag  = total_real * term_imag + total_imag * term_real;  
    }
    
    //Computes: (2*pi*i)^(nderivs) * (total_real + total_imag*i)
    double pi_mult = pow(2*M_PI, nderivs);
    /*Determines what the result of i^nderivs is, and performs the 
      correct multiplication afterwards.*/
    if (nderivs % 4 == 0) {
        dpr[0] = pi_mult*total_real;
        dpi[0] = pi_mult*total_imag;
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
//void integer_points(double* intpoints, 
//	       double* T,
//	       double* c
//	       int g, 
//	       double R)
//{
// int x, a, b;
//
//  struct slice {
//    double *vector;
//    double *center;
//    int g;
//    int R;
//  };
  //Initialize stack of points
  //Initialze stack
  //add (), g, c, R struct to stack
//  first = (double*)malloc(g*sizeof(double));
//  struct slice starter{first, c, g, R};
  //while we have elements in the stack
  //pop an element off the stack
  //calculate a and b
  //check if we've reached the end of the ellipsoid (a < b ?)
  /*If we have reached the end jump to the next iteration of loop, if in addition
    the stack is empty we have to immediately quit the loop */
  //if g == 0, append the points
  //Otherwise, calculate the starting point for the next dimension down,
  //append that information to the top of the stack
  
  //If the stack is empty stop
  
//  points = (double*)malloc( * sizeof(double))
  
//}


#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif /* __RIEMANNTHETA_H__ */
