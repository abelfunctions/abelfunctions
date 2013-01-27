"""
Grady Williams
January 3, 2013

Parallel program for computing the value of the Riemann Theta Function at 
multiple points. Returns a list of complex values.
"""

"""
Prepares the variables for computation on a GPU,

X = real part of the Omega matrix
Yinv = The inverser of the imaginary part of the Omega matrix
T = Cholesky Decomposition of Omega
Z = List of points to compute the function at
S = List of integer points to sum over
g = genus
"""

import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

def compute(X, Yinv, T, Z, S, g):
    S = np.array(S)
    x = Z.real
    y = Z.imag
    
    #Put all arrays into contiguous memory
    X = np.require(X, dtype = np.double, requirements=['A','W','O','C'])
    Yinv = np.require(Yinv, dtype = np.double, requirements=['A','W','O','C'])
    T = np.require(T, dtype = np.double, requirements=['A','W','O','C'])
    S = np.require(S, dtype = np.double, requirements=['A','W','O','C'])
    x = np.require(x, dtype = np.double, requirements=['A','W','O','C'])
    y = np.require(y, dtype = np.double, requirements=['A','W','O','C'])

    #Number of integer points to sum over
    N = S.size/g
    #Number of points to calculate the function at
    K = Z.size/g
    
    X_len = N
    Y_len = K
    
    #Create empty array to hold computed real values
    #First determine how many bytes we need to hold N complex values
    fsum_real = np.zeros(N*K)
    #Create empty array to hold computed imaginary values
    fsum_imag = np.zeros(N*K)

    #Allocate memory on the GPU
    fsum_reald = cuda.mem_alloc(fsum_real.nbytes)
    fsum_imagd = cuda.mem_alloc(fsum_imag.nbytes)
    Xd = cuda.mem_alloc(X.nbytes)
    Yinvd = cuda.mem_alloc(Yinv.nbytes)
    Td = cuda.mem_alloc(T.nbytes)
    xd = cuda.mem_alloc(x.nbytes)
    yd = cuda.mem_alloc(y.nbytes)
    Sd = cuda.mem_alloc(S.nbytes)
    
    #Transfer data from CPU to GPU
    cuda.memcpy_htod(fsum_reald, fsum_real)
    cuda.memcpy_htod(fsum_imagd, fsum_imag)
    cuda.memcpy_htod(Xd, X)
    cuda.memcpy_htod(Yinvd, Yinv)
    cuda.memcpy_htod(Td, T)
    cuda.memcpy_htod(xd, x)
    cuda.memcpy_htod(yd, y)
    cuda.memcpy_htod(Sd, S)
    
    #Prepare the first kernel for execution
    TILEHEIGHT = 32
    TILEWIDTH = 16
    partial_sums = func1(TILEWIDTH, TILEHEIGHT, g)
    reduction = func2()
    BLOCKSIZE = (TILEWIDTH, TILEHEIGHT, 1)
    GRIDSIZE = (N//TILEWIDTH + 1,K//TILEHEIGHT + 1, 1)

    #Make all scalars into numpy data types
    N = np.int32(N)
    K = np.int32(K)
    g = np.int32(g)

    partial_sums(fsum_reald, fsum_imagd, Xd, Yinvd, Td, xd, yd, Sd, g, N, K,
         block = BLOCKSIZE,
         grid = GRIDSIZE
         )

    cuda.Context.synchronize()
    
    """
    Now we perform GPU sum reduction on the imaginary and real parts
    """ 
    BLOCKSIZE = 128
    out_real = cuda.mem_alloc(fsum_real.nbytes)
    out_imag = cuda.mem_alloc(fsum_imag.nbytes)
    stride = np.int32(X_len)

    while (X_len > 1):
        J = (X_len - 1)//BLOCKSIZE + 1
        GRIDSIZE = (J, Y_len, 1)
        reduction(fsum_reald, out_real, np.int32(X_len), stride, 
                  block = (BLOCKSIZE,1,1), grid = GRIDSIZE)
        reduction(fsum_imagd, out_imag, np.int32(X_len), stride,
                  block = (BLOCKSIZE,1,1), grid = GRIDSIZE)
        X_len = J
        cuda.Context.synchronize()

        temp = fsum_reald
        fsum_reald = out_real
        out_real = temp

        temp = fsum_imagd
        fsum_imagd = out_imag
        out_imag = temp

    cuda.memcpy_dtoh(fsum_real, fsum_reald)
    cuda.memcpy_dtoh(fsum_imag, fsum_imagd)
    
    fsums = fsum_real[:K] + fsum_imag[:K]*1.0j
    return fsums

def func1(TILEWIDTH, TILEHEIGHT, g):
    template = """

/****************************************************************************

#include <stdlib.h>
#include <stdio.h>
#include <math.h>


normpart
--------

A helper function for the finite sum functions. Computes:

-pi * ||T*(n + fracshift)||^2

= -pi * ||T * (n + (shift - intshift))||^2

= -pi * ||T * (n + Yinv*y - round(Yinv*y))||^2

***************************************************************************/

__device__ double normpart(double* Sd, double* Td, int g, double* Yinvd, 
			  double* yd, int z_start, int n_start)
{
  double norm = 0;
  int i,j,k;
  for (i = 0; i < g; i++) {
    double sum = 0;
    for (j = 0; j < g; j++) {
      double T_ij = Td[i*g + j];
      double n_i = Sd[n_start + i];
      double shift_j = 0;
      for (k = 0; k < g; k++) {
	shift_j += Yinvd[g*j + k]*yd[z_start + k];
      }
      sum += T_ij * (n_i + shift_j - round(shift_j));
    }
    norm += sum * sum;
  }
  return -M_PI * norm;
}

/*************************************************************************
exppart
-------

A helper function for the finite sum functions. Computes:

2pi * <(n - intshift), (1/2)X(n - intshift) + x>

=2pi * <n - round(shift), (1/2)X(n - round(shift) + x>

=2pi * <n - round(Yinv*y), (1/2)X(n - round(Yinv*y) + x>



***************************************************************************/

__device__ double exppart(double* Sd, double* Xd, double* xd, int g,
			 double *Yinvd, double* yd, int z_start,
			 int n_start)
{
  double exppart = 0;
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
    double B = Xshift_i + xd[z_start + i];
    exppart += A * B;
  }
  return 2 * M_PI * exppart;
}

/***********************************************************************

Kernel Function

************************************************************************/
__global__ void riemann_theta(double* fsum_reald, double* fsum_imagd,
		      double* Xd, double* Yinvd,
		      double* Td, double* xd, double* yd, double* Sd,
		      int g, int N, int K)
{
  /*Built in variables to be used, br is block row, bc is
  block column, and similiarly for tr and tc.*/
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  __shared__ double Sd_s[%d];
  __shared__ double xd_s[%d];
  __shared__ double yd_s[%d];

  int TILEWIDTH = %d;
  int TILEHEIGHT = %d;

  /*Determine n_1, the start of the summation vector,
  the full vector is of the form n_1, n_2, ..., n_g*/
  int n_start = (bx * TILEWIDTH + tx) * g;
  /*Now n = S[n_start], S[n_start + 1], ..., S[n_start + (g - 1)]*/

  /*Determine z the point of evaluation*/
  int z_start = (by * TILEHEIGHT + ty) * g;
  /*Now x = (x[z_start], x[z_start + 1], ... , x[z_start + (g-1)],
  and similiarly for y.*/

  if (n_start < N*g && z_start < K*g) {
    /*Compute the "cosine" and "sine" parts of the summand*/
    double ept, npt, cpt, spt;
    ept = exppart(Sd, Xd, xd, g, Yinvd, yd, z_start, n_start);
    npt = exp(normpart(Sd, Td, g, Yinvd, yd, z_start, n_start));
    cpt = npt * cos(ept);
    spt = npt * sin(ept);
  
    fsum_reald[n_start/g + z_start/g * N] = cpt;
    fsum_imagd[n_start/g + z_start/g * N] = spt;
  }
}


""" %(g*TILEWIDTH, g*TILEHEIGHT, g*TILEHEIGHT, TILEWIDTH, TILEHEIGHT)
    mod = SourceModule(template)
    return mod.get_function("riemann_theta")
    
def func2():
    mod = SourceModule("""
__global__ void reduction_kernel(double *A_d, double *A_outd, int num_elements, int stride)
{
  int tdx = threadIdx.x;
  int bdx = blockIdx.x;
  int bdy = blockIdx.y;
  int index = blockIdx.x * 128 + tdx;
  int row = bdy * stride;

  __shared__ double data[128];
  data[tdx] = 0;

  if (index < num_elements) {
    data[tdx] = A_d[row + index];
  }
  __syncthreads();

  for (int i = 1; i < 128; i *= 2) {
    if (tdx % (2*i) == 0) {
      data[tdx] += data[tdx + i];
    }
    __syncthreads();
  }

  if (tdx == 0) {
    A_outd[bdx + row] = data[0];
  }

  if (gridDim.x == 1) {
    A_outd[bdy] = data[0];
  }
}

""")

    return mod.get_function("reduction_kernel")

