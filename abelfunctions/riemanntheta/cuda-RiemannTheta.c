/*CUDA function for calculating multiple values of the riemann-theta function*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define TILEWIDTH = 16
#define TILEHEIGHT = 16

void Riemann(double* fsum_real, double* fsum_imag, double* X, double* Yinv,
	    double* T, double* x, double* y, double* S, int g, int N, int K)
{
  double* fsum_reald;
  double* fsum_imagd;
  double* helperd;
  double* Xd;
  double* Yinvd;
  double* Td;
  double* xd;
  double* yd;
  double* Sd;
  //Size for X, T, and Yinv
  int size1 = g*g*sizeof(double);
  //Size for x and y
  int size2 = K*g*sizeof(double);
  //Size for S
  int size3 = N*g*sizeof(double);
  //Size for fsum_imagd and fsum_reald
  int size4 = K*N*sizeof(double);

  //Step 1, transfer X, Yinv, T, x, y, and S to device memory:

  //(1) Transfer X
  cudaMalloc((void**) &Xd, size1);
  cudaMemcpy(Xd, X, size1, cudaMemcpyHostToDevice);
  //(2) Transfer Yinv
  cudaMalloc((void**) &Yinvd, size1);
  cudaMemcpy(Yinvd, Yinv, size1, cudaMemcpyHostToDevice);
  //(3) Transfer T
  cudaMalloc((void**) &Td, size1);
  cudaMemcpy(Td, T, size1, cudaMemcpyHostToDevice);
  //(4) Transfer x
  cudaMalloc((void**) &xd, size2);
  cudaMemcpy(xd, x, size2, cudaMemcpyHostToDevice);
  //(5) Transfer y
  cudaMalloc((void**) &yd, size2);
  cudaMemcpy(yd, y, size2, cudaMemcpyHostToDevice);
  //(6) Transfer S
  cudaMalloc((void**) &Sd, size3);
  cudaMemcpy(Sd, S, size3, cudaMemcpyHostToDevice);

  //Step 2, Initialize fsum_real, fsum_imag, and a helper array onto
  //the device
  cudaMalloc((void**) &fsum_reald, size4);
  cudaMalloc((void**) &fsum_imagd, size4);

  //Step 3, Kernel Invocation
  dim3 dimGrid(K/TILEHEIGHT + 1, N/TILEWIDTH + 1);
  dim3 dimBlock(TILEHEIGHT, TILEWIDTH);
  Kernel<<<dimGrid, dimBlock>>>(fsum_reald, fsum_imagd,
				helperd, Xd, Yinvd,
				Td, xd, yd, Sd, g, N, K);
  
  //Step 4, Transfer fsum_reald and fsum_imagd from device to host
  cudaMemcpy(fsum_real, fsum_reald, size4, cudaMemcpyDeviceToHost);
  cudaMemcpy(fsum_imag, fsum_imagd, size4, cudaMemcpyDeviceToHost);
  
  //Step 5, Free CUDA arrays
  cudaFree(fsum_reald);
  cudaFree(fsum_imagd);
  cudaFree(helperd);
  cudaFree(Xd);
  cudaFree(Yinvd);
  cudaFree(Td);
  cudaFree(xd);
  cudaFree(yd);
  cudaFree(Sd);

  //ALl Done
}

__global__void Kernel(double* fsum_reald, double* fsum_imagd,
		      double* Xd, double* Yinvd,
		      double* Td, double* xd, double* yd, double* Sd
		      int g, int N, int K)
{
  //Built in variables to be used, br is block row, bc is
  //block column, and similiarly for tr and tc.
  int bx = blockID.x;
  int by = blockID.y;
  int tx = threadID.x;
  int ty = threadID.y;

  //Determine n1, the start of the summation vector,
  //the full vector is of the form n1, n2, ..., ng
  int n_start = (bx * TILEWIDTH + tx) * g;
  //Now n = S[n_start], S[n_start + 1], ..., S[n_start + (g - 1)]
  
  //Determine z the point of evaluation
  int z_start = (by * TILEHEIGHT + ty) * g;
  //Now x = (x[z_start], x[z_start + 1], ... , x[z_start + (g-1)],
  //and similiarly for y.
  
  //Compute the "cosine" and "sine" parts of the summand
  double ept, npt, cpt, spt;
  ept = exppart(Sd, Xd, x, g, Yinvd, yd, z_start, n_start);
  npt = exp(normpart(Sd, Td, g, Yinvd, yd, z_start, n_start));
  cpt = npt * cos(ept);
  spt = npt * sin(ept);
  
  *fsum_reald[n_start + N * z_start] = cpt;
  *fsum_imagd[n_start + N * z_start] = spt;
  
}


/****************************************************************************
normpart
--------

A helper function for the finite sum functions. Computes:

-pi * ||T*(n + fracshift)||^2

= -pi * ||T * (n + (shift - intshift))||^2

= -pi * ||T * (n + Yinv*y - round(Yinv*y))||^2

***************************************************************************/

__device__double normpart(double* Sd, double* Td, int g, double* Yinvd, 
			  double* yd, int z_start, int n_start)
{
  double norm = 0;
  int i,j,k;
  for (i = 0; i < g; i++) {
    double sum = 0;
    for (j = 0; j < g; j++) {
      double T_ij = Td[i + j*g];
      double n_j = Sd[n_start + j];
      double shift_j = 0;
      for (k = 0; k < g; k++) {
	shift_j += Yinv[k + g*j]*yd[z_start + k];
      }
      sum += T_ij * (n_j + shift_j - round(shift_j));
    }
    norm += sum * sum;
  }
  return -M_PI * norm;
}

/****************************************************************************
exppart
-------

A helper function for the finite sum functions. Computes:

2pi * <(n - intshift), (1/2)X(n - intshift) + x>

=2pi * <n - round(shift), (1/2)X(n - round(shift) + x>

=2pi * <n - round(Yinv*y), (1/2)X(n - round(Yinv*y) + x>



***************************************************************************/

__device__double exppart(double* Sd, double* Xd, double* xd, int g,
			 double *Yinvd, double* yd, int z_start,
			 int n_start)
{
  double exxpart = 0;
  int i,j,k,h;
  for (i = 0; i < g; i++) {
    double n_i = Sd[n_start + i];
    double shift_i = 0;
    for (k = 0; k < g; k++) {
      shift_i += Yinvd[k + i*g] * yd[z_start + k];
    }
    double A = n_i - round(shift_i);
    double Xshift_i = 0;
    for (j = 0; j < g; j++) {
      double X_ij = Xd[j + i * g];
      double shift_j = 0;
      for (h = 0; h < g; h++) {
	shift_j += Yinvd[h + j * g] * yd[z_start + h];
      }
      Xshift_i += (.5) * (X_ij * (Sd[n_start + j] - round(shift_j)));
    }
    double B = Xshift_i + x[z_start + i];
    exppart += A * B;
  }
  return 2 * M_PI * exppart;
}
   
  
 
