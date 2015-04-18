/*
  finite_sum_opencl.cl
  
  OpenCL code for computing the finite sum in the computation of the Riemann
  theta function.
*/


/*
 */
#define PI 3.141592653589793

float exppart(int, __constant const float*, __constant const float*, 
	      __constant const float*, const int,  __constant const float*);

float normpart(int, __constant const float*, __constant const float*, const int, 
	       __constant const float*);

/*
  Given integer vector S[i], compte the exppart of the sum.
*/
float exppart(int i,
	      __constant const float* X,
	      __constant const float* intshift, 
	      __constant const float* x,
	      const int g,
	      __constant const float* S)
{
  // perform dot product of matrix mult
  float sum = 0;
  float v_j, Xw_j;
  for (int j=0; j<g; j++) {
    // jth component of a-intshift
    v_j = S[i*g+j]-intshift[j];

    // jth component of (1/2) X*(a-intshift) + x
    Xw_j = 0;
    for (int k=0; k<g; k++) {
      Xw_j += 0.5 * X[j*g+k] * (S[i*g+k]-intshift[k]);
    }
    Xw_j += x[j];

    // add the jth component to the dot product container
    sum += v_j*Xw_j;
  }
  
  return 2*PI*sum;
}

/*
  Given integer vector S[i], compte the exppart of the sum.
*/
float normpart(int i, 
	       __constant const float* T,
	       __constant const float* fracshift,
	       const int g,
	       __constant const float* S)
{
  float sum = 0;
  float v_j;

  // loop over elements of two vectors
  for (int j=0; j<g; j++){
    // calculate jth component of v=T(a+fracshift)
    v_j = 0;
    for (int k=0; k<g; k++) {
      v_j += T[j*g+k] * (S[i*g+k]+fracshift[k]);
    }

    sum += v_j*v_j;
  }

  return -PI*sum;
}



__kernel void 
finite_sum_without_derivs(__constant const float* X,
			  __constant const float* T,
			  __constant const float* x,
			  __constant const float* intshift,
			  __constant const float* fracshift,
			  const int g,
			  __constant const float* S,
			  __global float* fsum_real,   // accum vectors
			  __global float* fsum_imag)   // accum vectors
{
  float ept, npt, cpart, spart;
  int i;
  i = get_global_id(0);

  // compute exponential and norm parts
  ept = exppart(i,X,intshift,x,g,S);
  npt = exp(normpart(i,T,fracshift,g,S));

  // split up into real and imaginary (cosine and sine) parts
  cpart = npt * cos(ept);
  spart = npt * sin(ept);

  // add to real and imaginary parts of summation
  fsum_real[i] = cpart;
  fsum_imag[i] = spart;
}
