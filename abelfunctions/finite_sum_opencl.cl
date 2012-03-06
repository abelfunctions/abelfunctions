/*
  finite_sum_opencl.cl
  
  OpenCL code for computing the finite sum in the computation of the Riemann
  theta function.
*/


/*
 */
#define PI 3.141592653589793

float exppart(int, __global float*, __global float*, __global float*, int, 
	      __global float*, int);

float normpart(int, __global float*, __global float*, int,
	       __global float* int);

/*
  Given integer vector S[i], compte the exppart of the sum.
*/
float exppart(int i,
	      __global float* X,
	      __global float* intshift, 
	      __global float* x,
	      int N,
	      __global float* S,
	      int L)
{
  // perform dot product of matrix mult
  float sum = 0;
  float v_j, Xw_j;
  for (int j=0; j<N; j++) {
    // jth component of a-intshift
    vj = S[i*N+j]-intshift[j];

    // jth component of (1/2) X*(a-intshift) + x
    Xw_j = 0;
    for (int k=0; k<N; k++) {
      Xw_j += 0.5 * X[j + N*k] * (S[i*N+k]-intshift[k]) + x[j];
    }

    // add the jth component to the dot product container
    sum += v_j*Xw_j
  }
  
  // scale and return
  sum *= 2*PI;
  return sum
}

/*
  Given integer vector S[i], compte the exppart of the sum.
*/
float normpart(float* a,
	       __global float* T,
	       __global float* fracshift,
	       int N,
	       __global float* S,
	       int L);
{

}



__kernel void finite_sum(__global float* X,
			 __global float* Yinv,
			 __global float* T,
			 __global float* x,
			 __global float* y, 
			 int N
			 __global float* S,
			 int L)
{
  int i = get_global_id(0);   // index of the integer point
}
