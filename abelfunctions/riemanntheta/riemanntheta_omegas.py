"""
Grady Williams
May 28, 2013

Parallel program for computing the Riemann Theta function at a 
single point for multiple values of Omega.

Input -> A point to be computed at and a list of Omega Matrices in tuple
         form (X, Yinv, T) where X is the real part of Omega, Yinv is the
         inverse of the imaginary part, and T is the upper triangular 
         cholesky decomposition of the imaginary part.

Output -> A list of complex values

x = real part of the point to be computed at

y = imaginary part of the point to be computed at

S = list of integer points to sum over.

g = genus of the riemann matrix, the program assumes that all 
    Omegas have the same matrix.

A 'd' suffix means that the data structure is on the GPU device. e.g Sd is 
the list of integer points stored on the device.

A 's' suffix implies plurality
"""

import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray

class RiemannThetaOmegas:

    """
    Initializes the riemanntheta_omega object. The main job of __init__ is to
    compile and store the cuda function which will be needed later.
    """
    def __init__(self, tileheight, tilewidth):
        self.tileheight = 8
        self.tilewidth = 64
        #Declares global variables to be determined later
        self.xd = None
        self.yd = None
        self.Sd = None
        self.g = None
        self.finite_sum_without_derivs = None
        self.finite_sum_with_derivs = None
        self.reduction = self.func2()

    """
    Compiles func1() based on g. Note that this function is called every time 
    that g changes
    """
    def compile(self, g):
        self.g=g
        (self.finite_sum_without_derivs, self.finite_sum_with_derivs, 
        self.xd, self.yd) = self.func1(g, self.tilewidth,self.tileheight)

    """
    Stores the list of integer points as a gpuarray
    """
    def cache_intpoints(self, S, gpu_already = False):
        if (not gpu_already):
            S = np.require(S, dtype = np.double, requirements=['A','W','O','C'])
            self.Sd = gpuarray.to_gpu(S)
        else:
            self.Sd = S

    """
    Stores the point for the Omegas to be computed at as a gpuarray
    """
    def cache_z(self, z):
        x = np.require(z.real, dtype = np.double, requirements = ['A','W','O','C'])
        y = np.require(z.imag, dtype = np.double, requirements = ['A','W','O','C'])
        xd = gpuarray.to_gpu(x)
        yd = gpuarray.to_gpu(y)
        cuda.memcpy_dtod(self.xd, xd.ptr, xd.nbytes)
        cuda.memcpy_dtod(self.yd, yd.ptr, yd.nbytes)

    def compute_v_without_derivs(self, Xs, Yinvs, Ts):
        #Turn the parts of omega into gpuarrays
        Xs = np.require(Xs, dtype = np.double, requirements=['A', 'W', 'O', 'C'])
        Yinvs = np.require(Yinvs, dtype = np.double, requirements=['A', 'W', 'O', 'C'])
        Ts = np.require(Ts, dtype = np.double, requirements=['A', 'W', 'O', 'C'])
        Xs_d = gpuarray.to_gpu(Xs)
        Yinvs_d = gpuarray.to_gpu(Yinvs)
        Ts_d = gpuarray.to_gpu(Ts)
        #Determine N = the number of integer points to sum over
        #          K = the number of different omegas to compute the function at
        N = self.Sd.size/self.g
        K = Xs.size/(self.g**2)
        #Create room on the gpu for the real and imaginary finite sum calculations
        fsum_reald = gpuarray.zeros(N*K, dtype=np.double)
        fsum_imagd = gpuarray.zeros(N*K, dtype=np.double)
        #Turn all scalars into numpy data types
        Nd = np.int32(N)
        Kd = np.int32(K)
        gd = np.int32(self.g)
        blocksize = (self.tilewidth, self.tileheight, 1)
        gridsize = (N//self.tilewidth + 1, K//self.tileheight + 1, 1)
        self.finite_sum_without_derivs(fsum_reald, fsum_imagd, Xs_d, Yinvs_d, Ts_d,
                                       self.Sd, gd, Nd, Kd,
                                       block = blocksize,
                                       grid = gridsize)
        cuda.Context.synchronize()
        fsums_real = self.sum_reduction(fsum_reald, N, K, Kd, Nd)
        fsums_imag = self.sum_reduction(fsum_imagd, N, K, Kd, Nd)
        return fsums_real + 1.0j*fsums_imag

    def compute_v_with_derivs(self, Xs, Yinvs, Ts, derivs):
        #Turn the parts of omega into gpuarrays
        Xs = np.require(Xs, dtype = np.double, requirements=['A', 'W', 'O', 'C'])
        Yinvs = np.require(Yinvs, dtype = np.double, requirements=['A', 'W', 'O', 'C'])
        Ts = np.require(Ts, dtype = np.double, requirements=['A', 'W', 'O', 'C'])
        Xs_d = gpuarray.to_gpu(Xs)
        Yinvs_d = gpuarray.to_gpu(Yinvs)
        Ts_d = gpuarray.to_gpu(Ts)
        #Turn the set of derivatives into gpuarrays
        deriv_real = np.require(derivs.real, dtype = np.double, requirements=['A','W','O', 'C'])
        deriv_imag = np.require(derivs.imag, dtype = np.double, requirements=['A', 'W', 'O', 'C'])
        nderivs = len(derivs)
        deriv_reald = gpuarray.to_gpu(deriv_real)
        deriv_imagd = gpuarray.to_gpu(deriv_imag)
        nderivs_d = np.int32(nderivs)
        #Determine N = the number of integer points to sum over
        #          K = the number of different omegas to compute the function at
        N = self.Sd.size/self.g
        K = Xs.size/(self.g**2)
        #Create room on the gpu for the real and imaginary finite sum calculations
        fsum_reald = gpuarray.zeros(N*K, dtype=np.double)
        fsum_imagd = gpuarray.zeros(N*K, dtype=np.double)
        #Turn all scalars into numpy data types
        Nd = np.int32(N)
        Kd = np.int32(K)
        gd = np.int32(self.g)
        blocksize = (self.tilewidth, self.tileheight, 1)
        gridsize = (N//self.tilewidth + 1, K//self.tileheight + 1, 1)
        self.finite_sum_with_derivs(fsum_reald, fsum_imagd, Xs_d, Yinvs_d, Ts_d, 
                                    self.Sd, deriv_reald, deriv_imagd, 
                                    nderivs, gd, Nd, Kd,
                                    block = blocksize,
                                    grid = gridsize)
        cuda.Context.synchronize()
        fsums_real = self.sum_reduction(fsum_reald, N, K, Kd, Nd)
        fsums_imag = self.sum_reduction(fsum_imagd, N, K, Kd, Nd)
        return fsums_real + 1.0j*fsums_imag
        
        
    def sum_reduction(self, fsum, N, K, Kd, Nd):
        out = gpuarray.zeros(K*N, dtype = np.double)
        blockheight = self.tileheight
        blockwidth = self.tilewidth
        while (N > 1):
            J = (N - 1)//blockwidth + 1
            gridsize = (J, (K-1)//blockheight + 1, 1)
            self.reduction(fsum, out, Nd, Kd, 
                      block = (blockwidth,blockheight,1), grid = gridsize)
            N = J
            Nd = np.int32(N)
            cuda.Context.synchronize()
            temp = fsum
            fsum = out
            out = temp
        #Get the real and imaginary parts from the GPU
        fsum_final = fsum.get()
        #We only care about the first K elements since 
        #that's where the summation function puts the 
        #values.
        fsum_final = fsum_final[:K]
        return fsum_final

    def func1(self, g, TILEWIDTH, TILEHEIGHT):
        template = """
        #include <stdlib.h>
        #include <stdio.h>
        #include <math.h>
        
        #define GENUS %d
        #define TILEHEIGHT %d
        #define TILEWIDTH %d

        __device__ __constant__ double xd[GENUS];
        __device__ __constant__ double yd[GENUS];
        
        /***************************************************************************

        normpart
        --------

        A helper function for the finite sum functions. Computes:

        -pi * ||T*(n + fracshift)||^2

        = -pi * ||T * (n + (shift - intshift))||^2

        = -pi * ||T * (n + Yinv*y - round(Yinv*y))||^2

        ***************************************************************************/
        
        __device__ double normpart(int g, double* Yinvd_s, double* Td_s, double* Sd_s)
        {
          int tx = threadIdx.x;
          int ty = threadIdx.y;
          double norm = 0;
          int i,j,k;
          for (i = 0; i < g; i++) {
            double sum = 0;
            for (j = 0; j < g; j++) {
              double T_ij = Td_s[ty*g*g + i*g + j];
              double n_j = Sd_s[tx*g + j];
              double shift_j = 0;
              for (k = 0; k < g; k++) {
                shift_j += Yinvd_s[ty*g*g + g*j + k]*yd[k];
              }
            sum += T_ij * (n_j + shift_j - round(shift_j));
            }
          norm += sum*sum;
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
        
        __device__ double exppart(int g, double *Xd_s, double *Yinvd_s, double* Sd_s)
        {
          int tx = threadIdx.x;
          int ty = threadIdx.y;
          double exppart = 0;
          int i,j,k,h;
          for (i = 0; i < g; i++) {
            double n_i = Sd_s[tx*g + i];
            double shift_i = 0;
            for (k = 0; k < g; k++) {
              shift_i += Yinvd_s[ty*g*g + i*g + k] * yd[k];
            }
            double A = n_i - round(shift_i);
            double Xshift_i = 0;
            for (j = 0; j < g; j++) {
              double X_ij = Xd_s[ty*g*g + i*g + j];
              double shift_j = 0;
              for (h = 0; h < g; h++) {
                shift_j += Yinvd_s[ty*g*g + j*g + h] * yd[h];
              }
              Xshift_i += .5 * (X_ij * (Sd_s[tx*g + j] - round(shift_j)));
            }
            double B = Xshift_i + xd[i];
            exppart += A*B;
          }
          return 2* M_PI * exppart;
        }

        /****************************************************************************
        Derivative Product
        
        Computes:
        
           ___
           | | 
           | | 2*pi*I <d, n - intshift>
        d in derivs
        ****************************************************************************/
        __device__ void deriv_prod(int g, double *Sd_s, double* Yinvd_s,
                                   double* dpr, double* dpi, double* deriv_real, 
                                   double* deriv_imag, int nderivs)
        {
          int tx = threadIdx.x;
          int ty = threadIdx.y;
          double total_real = 1;
          double total_imag = 0;
          
          int i,j,k;
          for (i = 0; i < nderivs; i++) {
            double term_real = 0;
            double term_imag = 0;
            for (j = 0; i < g; i++) {
              double shift_j = 0;
              for (k = 0; k < g; k++) {
                shift_j += Yinvd_s[ty*g*g + j*g + k] * yd[k];
              }
              double intshift = round(shift_j);
              double nmintshift = Sd_s[tx*g + j] - intshift;
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
          if (nderivs %% 4 == 0) {
            dpr[0] = pi_mult*total_real;
            dpi[0] = pi_mult*total_imag;
          }
          else if (nderivs %% 4 == 1) {
            dpr[0] = -pi_mult * total_imag;
            dpi[0] = pi_mult * total_real;
          }
          else if (nderivs %% 4 == 2) {
            dpr[0] = -pi_mult * total_real;
            dpi[0] = -pi_mult * total_imag;
          }
          else if (nderivs %% 4 == 3) {
            dpr[0] = pi_mult * total_imag;
            dpi[0] = -pi_mult * total_real;
          }
        }


        /**************************************************************************
        
        Finite Sum Without Derivatives Kernel Function
        
        **************************************************************************/
        __global__ void riemann_theta(double *fsum_reald, double *fsum_imagd, double *Xd,
                                      double *Yinvd, double* Td, double *Sd, int g, int N, int K)
        {
          /*Built in variables to be used, x variable denotes the summation index
          while the y variable denotes the Omega index*/
          int bx = blockIdx.x;
          int by = blockIdx.y;
          int tx = threadIdx.x;
          int ty = threadIdx.y;
          
          __shared__ double Sd_s[TILEWIDTH * GENUS];
          __shared__ double Xd_s[TILEHEIGHT * GENUS * GENUS];
          __shared__ double Yinvd_s[TILEHEIGHT * GENUS * GENUS];
          __shared__ double Td_s[TILEHEIGHT * GENUS * GENUS]; 

          /*Determine n_0, the start of the summation vector,
          the full vector is of the form, n_0, n_1, n_2, n_g*/
          int n_start = (bx * TILEWIDTH + tx) * g;
          /* Now n = S[n_start], S[n_start + 1], S[n_start + 2]...S[n_start + (g-1)] */
        
          /*Determine the Omega to evaluate on*/
          int omega_start = (by*TILEHEIGHT + ty)*g*g;
          /* Now omega is Omega[omega_start], ... Omega[omega_start + g*g-1]
          where Omega is Xd, Yinvd, and Td */
          
          /*Load data into shared arrays */
          int i;
          for (i = 0; i < g; i++){
            Sd_s[tx*g + i] = Sd[n_start + i];
          }
          for (i = 0; i < g*g; i++) {
            Xd_s[ty*g*g + i] = Xd[omega_start + i];
            Yinvd_s[ty*g*g + i] = Yinvd[omega_start + i];
            Td_s[ty*g*g + i] = Td[omega_start + i];
          }

          __syncthreads();
          
          if (n_start < N*g && omega_start < K*g*g) {
            /*Compute the 'cosine' and 'sine' parts of the summand*/
            double ept, npt, cpt, spt;
            ept = exppart(g, Xd_s, Yinvd_s, Sd_s);
            npt = exp(normpart(g, Yinvd_s, Td_s, Sd_s));
            cpt = npt*cos(ept);
            spt = npt*sin(ept);

            fsum_reald[n_start/g + omega_start/g/g * N] = cpt;
            fsum_imagd[n_start/g + omega_start/g/g * N] = spt;
          }
       }

       /***********************************************************************
       
       Finite Sum with Derivatives Kernel Function
       
       ************************************************************************/
       __global__ void riemann_theta_derivatives(double* fsum_reald, double* fsum_imagd, 
                                                 double* Xd, double *Yinvd, double *Td, 
                                                 double *Sd, double *deriv_reald, 
                                                 double *deriv_imagd, 
                                                 int nderivs, int g, int N, int K)
        {
           /*Built in variables to be used, x variable denotes the summation index
          while the y variable denotes the Omega index*/
          int bx = blockIdx.x;
          int by = blockIdx.y;
          int tx = threadIdx.x;
          int ty = threadIdx.y;
          
          __shared__ double Sd_s[TILEWIDTH * GENUS];
          __shared__ double Xd_s[TILEHEIGHT * GENUS * GENUS];
          __shared__ double Yinvd_s[TILEHEIGHT * GENUS * GENUS];
          __shared__ double Td_s[TILEHEIGHT * GENUS * GENUS]; 

          /*Determine n_0, the start of the summation vector,
          the full vector is of the form, n_0, n_1, n_2, n_g*/
          int n_start = (bx * TILEWIDTH + tx) * g;
          /* Now n = S[n_start], S[n_start + 1], S[n_start + 2]...S[n_start + (g-1)] */
          
          /*Determine the Omega to evaluate on*/
          int omega_start = (by*TILEHEIGHT + ty)*g*g;
          /* Now omega is Omega[omega_start], ... Omega[omega_start + g*g-1]
          where Omega is Xd, Yinvd, and Td */
          
          /*Load data into shared arrays */
          int i;
          for (i = 0; i < g; i++){
            Sd_s[tx*g + i] = Sd[n_start + i];
          }
          for (i = 0; i < g*g; i++) {
            Xd_s[ty*g*g + i] = Xd[omega_start + i];
            Yinvd_s[ty*g*g + i] = Yinvd[omega_start + i];
            Td_s[ty*g*g + i] = Td[omega_start + i];
          }

          __syncthreads();

          if (n_start < N*g && omega_start < K*g*g) {
            /*Compute the 'cosine' and 'sine' parts of the summand */
            double dpr[1];
            double dpi[1];
            dpr[0] = 0;
            dpi[0] = 0;
            double ept, npt, cpt, spt;            
            ept = exppart(g, Xd_s, Yinvd_s, Sd_s);
            npt = exp(normpart(g, Yinvd_s, Td_s, Sd_s));
            cpt = npt*cos(ept);
            spt = npt*sin(ept);
            deriv_prod(g,Sd_s,Yinvd_s,dpr,dpi, deriv_reald,deriv_imagd, nderivs);
            fsum_reald[n_start/g + omega_start/g/g * N] = dpr[0] * cpt - dpi[0] * spt;
            fsum_imagd[n_start/g + omega_start/g/g * N] = dpi[0] * cpt + dpr[0] * spt;
          }
        } 
        
       """ %(g, TILEHEIGHT, TILEWIDTH)
        mod = SourceModule(template)
        func = mod.get_function("riemann_theta")
        deriv_func = mod.get_function("riemann_theta_derivatives")
        xd = mod.get_global("xd")[0]
        yd = mod.get_global("yd")[0]
        return func, deriv_func, xd, yd
        

    def func2(self):
        template = """

   #define BLOCKWIDTH %d
   #define BLOCKHEIGHT %d

    __global__ void reduction_kernel(double *A_d, double *A_outd, int x_len, int y_len, int POINTS)
    {
      int tdx = threadIdx.x;
      int tdy = threadIdx.y;
      int bdx = blockIdx.x;
      int bdy = blockIdx.y;

      int x_ind = bdx*BLOCKWIDTH + tdx;
      int y_ind = bdy*BLOCKHEIGHT + tdy;

      __shared__ double data[BLOCKWIDTH * BLOCKHEIGHT];
      data[BLOCKWIDTH*tdy + tdx] = 0;

      if (x_ind < x_len && y_ind < y_len) {
        data[tdy*BLOCKWIDTH + tdx] = A_d[y_ind*x_len + x_ind];
      }
      __syncthreads();

      for (int i = BLOCKWIDTH/2; i > 0; i >>= 1) {
        if (tdx < i) {
          data[BLOCKWIDTH*tdy + tdx] += data[BLOCKWIDTH*tdy + tdx + i];
        }
        __syncthreads();
      }

      if (tdx == 0) {
        int newLength = (x_len - 1)/BLOCKWIDTH + 1;
        A_outd[y_ind*newLength + bdx] = data[BLOCKWIDTH*tdy];
      }

      if (gridDim.x == 1) {
        A_outd[y_ind] = data[BLOCKWIDTH*tdy];
      }
    }

    """%(self.tilewidth, self.tileheight)
        mod = SourceModule(template)
        return mod.get_function("reduction_kernel")
