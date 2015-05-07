/*=============================================================================

The LLL-Reduction Algorithm

Authors
-------

* Grady Williams (gradyrw@gmail.com)
* Chris Swierczewski (cswiercz@gmail.com)

=============================================================================*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/*
  gram_schmidt

  Numerically stable Gram-Schmidt algorithm.

  Given a set of `n` vectors `b_1, ..., b_n` construct a collection of
  orthogonal vectors `b*_1, ..., b*_n` spanning the same space. The vectors are
  stored as **columns** of b. THIS SHOULD BE CHANGED IN THE FUTURE TO FOLLOW
  STANDARD C PRACTICES FOR STORING VECTORS.

  Parameters
  ----------
  b : double[:]
      An array of `n` vectors each of size `n`.
  n : int

  Returns
  -------
  b : double[:]
      Orthogonalized vectors.
  mu : double[:]
      Matrix of orthogonalization parameters

          mu_ij = <bi, b*j> / <b*j, b*j>
  B : double[:]
      Square norms of orthogonalized vectors.
*/
void gram_schmidt(double* b, double* mu, double* B, int n)
{
  double numerator;
  double dot;
  int i,j,k;
  double* b_star = (double*)malloc(n*n*sizeof(double));
  double* temp = (double*)malloc(n*sizeof(double));

  for (i = 0; i < n; i++)
    {
      // (1) copy non-orthogonal vector: b*_i = b_i
      for (k = 0; k < n; k++)
	b_star[k*n + i] = b[k*n + i];

      // temp keeps track of the shift in the gram schmidt algorithm
      for (k = 0; k < n; k++)
	temp[k] = 0;

      // for each previously computed b*j perform the steps:
      //
      //   (2) mu_ij = <bi, b*j> / B[j]  (compute shift coefficient)
      //   (3) b*i = b*i - mu_ij*b*j     (shift by each previous b*j)
      //
      // note that this is not performed in the first iteration when i=0
      for (j = 0; j < i; j++)
	{
	  // (2) compute mu_ij = <b_i, b*_j> / <b*_j, b*_j>
	  numerator = 0;
	  for (k = 0; k < n; k++)
	    numerator += b[k*n + i] * b_star[k*n + j];
	  mu[i*n + j] = numerator / B[j];
	  //	  printf("\nnumerator = %f \t B[%d] = %f", numerator, j, B[j]);

	  // (3) shift b*i by - mu_ij b*j
	  for (k = 0; k < n; k++)
	    b_star[k*n + i] -= mu[i*n + j] * b_star[k*n + j];
	}

      // (4) store the dot product Bi = <b*i, b*i>
      dot = 0;
      for (k = 0; k < n; k++)
	dot += b_star[k*n + i]*b_star[k*n + i];
      B[i] = dot;
    }

  free(b_star);
  free(temp);
}


/*
  nearest_integer_shift

  Performs operation (*) on p. 521 of [LLL].

  Parameters
  ----------
  mu : double[:,:]
  b : double[:,:]
  k,l : int
  n : int
      Size of each vector.
  lc : double
      LLL parameter.
 */
void nearest_integer_shift(double* mu, double* b, int k, int l,
			   int n, double lc)
{
  int i,j;
  double r = round(mu[k*n + l]);

  if (fabs(mu[k*n+l]) > lc)
    {
      // shift bk by (-r*bl)
      for (i = 0; i < n; i++)
	b[i*n + k] -= r*b[i*n + l];

      // shift mu_kj by (-r*mu_lj) for j=0,...,l-2
      for (j = 0; j < l-1; j++)
	mu[k*n + j] -= r*mu[l*n + j];

      // shift mu_kl by (-r)
      mu[k*n + l] -= r;
    }
}


/*
  lll_reduce

  Performs Lenstra-Lenstra-Lovasv reduction on a given lattice n in
  n-dimensional real space. The input matrix b is in the usual C-ordering but
  the algorithm works on the columns. (This should be rewritten in a future
  update for performance purposes.)

  Parameters
  ----------
  b : double[:]
      Input array / `n x n` matrix.
  n : int
  lc,uc : double
      The LLL parameters.

  Returns
  -------
  b : double[:]
      The LLL reduction of the columns of the input `b`.
*/
void lll_reduce(double* b, int n, double lc, double uc)
{
  int i,j,k,l;
  double tmp, B_tmp, mu_tmp;
  double swap_condition;
  double* mu = (double*)calloc(n*n, sizeof(double));
  double* B = (double*)calloc(n, sizeof(double));

  // initialize mu and B with zeros
  for (i = 0; i < n*n; i++)
    mu[i] = 0;
  for (i = 0; i < n; i++)
    B[i] = 0;

  // orthogonalize the columns of b and obtain the scaling factors B and mu
  gram_schmidt(b,mu,B,n);
  k = 1;
  while (k < n)
    {
      nearest_integer_shift(mu, b, k, k-1, n, lc);
      swap_condition = (uc - mu[k*n + (k-1)]*mu[k*n + (k-1)])*B[k-1];
      if (B[k] < swap_condition)
	{
	  // set the "constant parameters" for this round
	  mu_tmp = mu[k*n + (k-1)];
	  B_tmp = B[k] + mu_tmp*mu_tmp*B[k-1];

	  // scale and swap mu and B values
	  mu[k*n + (k-1)] = mu_tmp*B[k-1] / B_tmp;
	  B[k] = B[k-1]*B[k] / B_tmp;
	  B[k-1] = B_tmp;

	  // swap b_(k-1) and b_k
	  for (i = 0; i < n; i++)
	    {
	      tmp = b[i*n + k];
	      b[i*n + k] = b[i*n + (k-1)];
	      b[i*n + (k-1)] = tmp;
	    }

	  // swap mu_(k-1),j and mu_k,j for j = 0,...,k-3
	  for (j = 0; j < k-2; j++)
	    {
	      tmp = mu[k*n + j];
	      mu[k*n + j] = mu[(k-1)*n + j];
	      mu[(k-1)*n + j] = tmp;
	    }

	  // perform the linear transformation for i = k,...,n-1
	  for (i = k; i < n; i++)
	    {
	      tmp = mu[i*n + (k-1)] - mu_tmp*mu[i*n + k];
	      mu[i*n + (k-1)] = mu[i*n + k] + mu[k*n + (k-1)] * tmp;
	      mu[i*n + k] = tmp;
	    }

	  if (k > 1)
	    k -= 1;
	}
      else
	{
	  // perform the integer shift for l = k-3, ..., 0
	  for (l = k-3; l >= 0; l--)
	    {
	      nearest_integer_shift(mu, b, k, l, n, lc);
	    }
	  k += 1;
	}
    }

  free(mu);
  free(B);
}
