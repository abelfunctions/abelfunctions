"""
Grady Williams
April 1, 2013

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

def compute_v(X, Yinv, T, Z, S, g, derivs = False, nderivs = 0, derivs = np.zeros(1)):
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
    if (derivs):
        deriv_real = np.require(derivs.real, dtype = np.double, requirements=['A','W','O', 'C'])
        deriv_imag = np.require(derivs.imag, dtype = np.double, requirements=['A', 'W', 'O', 'C'])
        nderivs = np.double(nderivs)

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
    xd = cuda.mem_alloc(x.nbytes)
    yd = cuda.mem_alloc(y.nbytes)
    Sd = cuda.mem_alloc(S.nbytes)
    if (derivs):
        deriv_reald = cuda.mem_alloc(deriv_real.nbytes)
        deriv_imagd = cuda.mem_alloc(deriv_imag.nbytes)
    
    #Prepare the first kernel for execution
    TILEHEIGHT = 32
    TILEWIDTH = 16
    partial_sums, partial_sums_derivs, Xd, Yinvd, Td  = func1(TILEWIDTH, TILEHEIGHT, g)
    reduction = func2()
    BLOCKSIZE = (TILEWIDTH, TILEHEIGHT, 1)
    GRIDSIZE = (N//TILEWIDTH + 1,K//TILEHEIGHT + 1, 1)

    #Transfer data from CPU to GPU
    cuda.memcpy_htod(fsum_reald, fsum_real)
    cuda.memcpy_htod(fsum_imagd, fsum_imag)
    cuda.memcpy_htod(Xd, X)
    cuda.memcpy_htod(Yinvd, Yinv)
    cuda.memcpy_htod(Td, T)
    cuda.memcpy_htod(xd, x)
    cuda.memcpy_htod(yd, y)
    cuda.memcpy_htod(Sd, S)

    #Make all scalars into numpy data types
    N = np.int32(N)
    K = np.int32(K)
    g = np.int32(g)
    
    if (not derivs):
        partial_sums(fsum_reald, fsum_imagd, xd, yd, Sd, g, N, K,
                     block = BLOCKSIZE,
                     grid = GRIDSIZE
                     )
    else:
        partial_sums_derivs(fsum_reald, fsum_imagd, deriv_reald, deriv_imagd, nderivs, xd, yd, Sd, g, N, K,
                     block = BLOCKSIZE,
                     grid = GRIDSIZE
                     )        
    cuda.Context.synchronize()
    
    """
    Now we perform GPU sum reduction on the imaginary and real parts
    """
    out_real = cuda.mem_alloc(fsum_real.nbytes)
    out_imag = cuda.mem_alloc(fsum_imag.nbytes)
    y_len = np.int32(Y_len)
    
    while (X_len > 1):
        J = (X_len - 1)//16 + 1
        GRIDSIZE = (J, (Y_len-1)//32 + 1, 1)
        reduction(fsum_reald, out_real, np.int32(X_len), y_len, 
                  block = (16,32,1), grid = GRIDSIZE)
        reduction(fsum_imagd, out_imag, np.int32(X_len), y_len,
                  block = (16,32,1), grid = GRIDSIZE)
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

def compute_u(z, Yinv, g):
    y = z.imag
    y_len = len(y)
    u = np.zeros(y_len)
    
    Yinv = np.require(Yinv, dtype = np.double, requirements = ['A', 'W', 'O', 'C'])
    y = np.require(y, dtype = np.double, requirements = ['A', 'W', 'O', 'C'])
    u = np.require(u, dtype = np.double, requirements = ['A', 'W', 'O', 'C'])

    yd = cuda.mem_alloc(y.nbytes)   
    ud = cuda.mem_alloc(u.nbytes)

    dotter, Yinvd = func3(g, 32)
    
    cuda.memcpy_htod(yd,y)
    cuda.memcpy_htod(ud,u)
    cuda.memcpy_htod(Yinvd, Yinv)
    
    Blocksize = (g, 32, 1)
    Gridsize = (1, (y_len - 1)//32 + 1,1)
    dotter(yd, ud, np.int32(g), np.int32(y_len), 
           block = Blocksize, grid = Gridsize)
    cuda.Context.synchronize()
    
    cuda.memcpy_dtoh(u, ud)
    return u

def func1(TILEWIDTH, TILEHEIGHT, g):
    template = """

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define GENUS %d
#define TILEHEIGHT %d
#define TILEWIDTH %d

__device__ __constant__ double Xd[GENUS*GENUS];
__device__ __constant__ double Yinvd[GENUS * GENUS];
__device__ __constant__ double Td[GENUS*GENUS];

/***************************************************************************

normpart
--------

A helper function for the finite sum functions. Computes:

-pi * ||T*(n + fracshift)||^2

= -pi * ||T * (n + (shift - intshift))||^2

= -pi * ||T * (n + Yinv*y - round(Yinv*y))||^2

***************************************************************************/

__device__ double normpart(int g, double* Sd_s, double* yd_s)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  double norm = 0;
  int i,j,k;
  for (i = 0; i < g; i++) {
    double sum = 0;
    for (j = 0; j < g; j++) {
      double T_ij = Td[i*g + j];
      double n_j = Sd_s[tx*g + j];
      double shift_j = 0;
      for (k = 0; k < g; k++) {
	shift_j += Yinvd[g*j + k]*yd_s[ty*g + k];
      }
      sum += T_ij * (n_j + shift_j - round(shift_j));
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

__device__ double exppart(int g, double* Sd_s, 
                          double* xd_s, double* yd_s)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  double exppart = 0;
  int i,j,k,h;
  for (i = 0; i < g; i++) {
    double n_i = Sd_s[tx*g + i];
    double shift_i = 0;
    for (k = 0; k < g; k++) {
      shift_i += Yinvd[k + i*g] * yd_s[ty*g + k];
    }
    double A = n_i - round(shift_i);
    double Xshift_i = 0;
    for (j = 0; j < g; j++) {
      double X_ij = Xd[j + i * g];
      double shift_j = 0;
      for (h = 0; h < g; h++) {
	shift_j += Yinvd[h + j * g] * yd_s[ty*g + h];
      }
      Xshift_i += (.5) * (X_ij * (Sd_s[tx*g + j] - round(shift_j)));
    }
    double B = Xshift_i + xd_s[ty*g + i];
    exppart += A * B;
  }
  return 2 * M_PI * exppart;
}

/**********************************************************************

Derivative Product

Computes: 
                   ___
                   | |    2*pi*I <d, n-intshift>
	           | |
	       d in derivs

=                  ___
                   | |    2*pi*I <d, n-round(shift)>
	           | |
	       d in derivs
=                  ___
                   | |    2*pi*I <d, n-round(Yinv*y)>
	           | |
	       d in derivs
************************************************************************/
__device__ double deriv_prod(int g, double* Sd_s, double* yd_s, double* dpr, double* dpi,
                             double* deriv_real, double* deriv_imag, int nderivs)
{
  int tdx = threadIdx.x;
  int tdy = threadIdx.y;

  double total_real = 1;
  double total_imag = 0;

  int i,j,k;
  for (i = 0; i < nderivs; i++){
    term real = 0;
    term imag = 0;
    for (j = 0; j < g; j++){
      double intshift_i = 0;
      for (k = 0; k < g; k++){
        shift += Yinvd[i*g + k] * yd_s[ty*g + k];
      }
      intshift = round(shift);
      nmintshift = Sd_s[tx*g + j] - intshift;
      term_real += deriv_real[j + g*i] * nmintshift;
      term_imag += deriv_imag[j + g*i] * nmintshift;
    }
  
    total_real = total_real * term_real - total_imag * term_imag;
    total_imag = total_real * term_imag + total_imag * term_real;
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



/***********************************************************************

Finite Sum Without Derivatives Kernel Function

************************************************************************/
__global__ void riemann_theta(double* fsum_reald, double* fsum_imagd,
		      double* xd, double* yd, double* Sd,
		      int g, int N, int K)
{
  /*Built in variables to be used, br is block row, bc is
  block column, and similiarly for tr and tc.*/
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  __shared__ double Sd_s[TILEWIDTH * GENUS];
  __shared__ double xd_s[TILEHEIGHT * GENUS];
  __shared__ double yd_s[TILEHEIGHT * GENUS];

  /*Determine n_1, the start of the summation vector,
  the full vector is of the form n_1, n_2, ..., n_g*/
  int n_start = (bx * TILEWIDTH + tx) * g;
  /*Now n = S[n_start], S[n_start + 1], ..., S[n_start + (g - 1)]*/

  /*Determine z the point of evaluation*/
  int z_start = (by * TILEHEIGHT + ty) * g;
  /*Now x = (x[z_start], x[z_start + 1], ... , x[z_start + (g-1)],
  and similiarly for y.*/

  /*Load data into shared arrays*/
  int i;
  for (i = 0; i < g; i++) {
    Sd_s[tx*g + i] = Sd[n_start + i];
    xd_s[ty*g + i] = xd[z_start + i];
    yd_s[ty*g + i] = yd[z_start + i];
  }

  __syncthreads();

  if (n_start < N*g && z_start < K*g) {
    /*Compute the "cosine" and "sine" parts of the summand*/
    double ept, npt, cpt, spt;
    ept = exppart(g,Sd_s, xd_s, yd_s);
    npt = exp(normpart(g, Sd_s, yd_s));
    cpt = npt * cos(ept);
    spt = npt * sin(ept);
  
    fsum_reald[n_start/g + z_start/g * N] = cpt;
    fsum_imagd[n_start/g + z_start/g * N] = spt;
  }
}

/************************************************************************************

Finite Sum with Derivatives Kernel Function

************************************************************************************/
__global__ void riemann_theta_derivatives(double* fsum_reald, double* fsum_imagd,
		      double* xd, double* yd, double* Sd, double* deriv_real,
		      double* deriv_imag, int nderivs, int g, int N, int K)
{
  /*Built in variables to be used, br is block row, bc is
  block column, and similiarly for tr and tc.*/
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  __shared__ double Sd_s[TILEWIDTH * GENUS];
  __shared__ double xd_s[TILEHEIGHT * GENUS];
  __shared__ double yd_s[TILEHEIGHT * GENUS];

  /*Determine n_1, the start of the summation vector,
  the full vector is of the form n_1, n_2, ..., n_g*/
  int n_start = (bx * TILEWIDTH + tx) * g;
  /*Now n = S[n_start], S[n_start + 1], ..., S[n_start + (g - 1)]*/

  /*Determine z the point of evaluation*/
  int z_start = (by * TILEHEIGHT + ty) * g;
  /*Now x = (x[z_start], x[z_start + 1], ... , x[z_start + (g-1)],
  and similiarly for y.*/

  /*Load data into shared arrays*/
  int i;
  for (i = 0; i < g; i++) {
    Sd_s[tx*g + i] = Sd[n_start + i];
    xd_s[ty*g + i] = xd[z_start + i];
    yd_s[ty*g + i] = yd[z_start + i];
  }

  __syncthreads();

  if (n_start < N*g && z_start < K*g) {
    /*Compute the "cosine" and "sine" parts of the summand*/
    double dpr[0];
    double dpi[0];
    dpr[0] = 0;
    dpi[0] = 0;
    double ept, npt, cpt, spt;
    ept = exppart(g,Sd_s, xd_s, yd_s);
    npt = exp(normpart(g, Sd_s, yd_s));
    cpt = npt * cos(ept);
    spt = npt * sin(ept);
    deriv_prod(g, Sd_s, yd_s, dpr, dpi, deriv_real, deriv_imag, nderivs);
    fsum_reald[n_start/g + z_start/g * N] = dpr[0] * cpt - dpi[0] * spt;
    fsum_imagd[n_start/g + z_start/g * N] = dpi[0] * cpt + dpr[0] * spt;
  }
}




""" %(g, TILEHEIGHT, TILEWIDTH)
    mod = SourceModule(template)
    func = mod.get_function("riemann_theta")
    deriv_func = mod.get_function("riemann_theta_derivatives")
    Xd = mod.get_global("Xd")[0]
    Yinvd = mod.get_global("Yinvd")[0]
    Td = mod.get_global("Td")[0]
    return (func, Xd, Yinvd, Td)
    
def func2():
    mod = SourceModule("""
__global__ void reduction_kernel(double *A_d, double *A_outd, int x_len, int y_len, int POINTS)
{
  int tdx = threadIdx.x;
  int tdy = threadIdx.y;
  int bdx = blockIdx.x;
  int bdy = blockIdx.y;

  int x_ind = bdx*16 + tdx;
  int y_ind = bdy*32 + tdy;

  __shared__ double data[16*32];
  data[16*tdy + tdx] = 0;

  if (x_ind < x_len && y_ind < y_len) {
    data[tdy*16 + tdx] = A_d[y_ind*x_len + x_ind];
  }
  __syncthreads();

  for (int i = 1; i < 16; i *= 2) {
    if (tdx % (2*i) == 0) {
      data[16*tdy + tdx] += data[16*tdy + tdx + i];
      
    }
    __syncthreads();
  }

  if (tdx == 0) {
    int newLength = (x_len - 1)/16 + 1;
    A_outd[y_ind*newLength + bdx] = data[16*tdy];
  }

  if (gridDim.x == 1) {
    A_outd[y_ind] = data[16*tdy];
  }
}

""")

    return mod.get_function("reduction_kernel")

def func3(g, TILEHEIGHT):
    template = """

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define GENUS %d
#define TILEHEIGHT %d

__device__ __constant__ double Yinvd[GENUS*GENUS]; 

__global__ void kernel(double* yd, double* u, int g, int y_len)
{
  int tdy = threadIdx.y;
  int bdy = blockIdx.y;
  int tdx = threadIdx.x;

  __shared__ double yd_s[GENUS*TILEHEIGHT];
  yd_s[tdy*g + tdx] = 0;
  if (bdy*TILEHEIGHT + tdy < y_len) {
    yd_s[tdy*g + tdx] = yd[(bdy*TILEHEIGHT + tdy) * g + tdx];
  }
  __syncthreads();

  if (bdy*TILEHEIGHT + tdy < y_len) {
    int i,j;
    double dot = 0;
    double Yinvy_i;
    for (i = 0; i < g; i++) {
      Yinvy_i = 0;
      for (j = 0; j < g; j++) {
        Yinvy_i += Yinvd[g*i + j] * yd_s[tdy*g+j];
      }
      dot += yd_s[tdy*g+i] * Yinvy_i;
    }
    u[bdy*TILEHEIGHT + tdy] = M_PI * dot;
  }
}
""" %(g, TILEHEIGHT)

    mod = SourceModule(template)
    func = mod.get_function("kernel")
    Yinvd = mod.get_global("Yinvd")[0]
    return (func, Yinvd)
