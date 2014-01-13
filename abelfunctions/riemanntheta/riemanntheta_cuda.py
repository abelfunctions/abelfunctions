"""
Grady Williams
April 1, 2013

Parallel program for computing the value of the Riemann Theta Function at 
multiple points. Returns a list of complex values.
"""

"""
Prepares the variables for computation on a GPU,

X = real part of the Omega matrix
Yinv = The inverse of the imaginary part of the Omega matrix
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
from pycuda import gpuarray

class RiemannThetaCuda:
    
    """
    Initializes the riemanntheta_cuda object. The main job of __init__ is to 
    compile and store the cuda functions which will be needed later.
    """
    def __init__(self, tileheight, tilewidth):
        self.tileheight = tileheight
        self.tilewidth = tilewidth
        #Declares global variables to be determined later
        self.finite_sum_without_derivs = None
        self.finite_sum_with_derivs = None
        self.Xd = None
        self.Td = None
        self.Yinv_vd = None
        self.Yinv_ud = None
        self.dot_prod = None
        self.Sd = None
        self.deriv_reald = None
        self.deriv_imagd = None
        self.g = None
        self.yd = None
        #Compiles a cuda sum-reduction function and stores it for later use
        self.reduction = self.grid_sum_reduction_d()
        

    """
    Compiles finite_sum_d() and exponential_growth_d() based on g. Note that this function is called every time 
o   that g changes
    """
    def compile(self, g):
        self.g=g
        (self.finite_sum_without_derivs, self.finite_sum_with_derivs,
         self.Xd, self.Yinv_vd, self.Td) = self.finite_sum_d(self.tilewidth, self.tileheight, g)
        self.dot_prod, self.Yinv_ud = self.exponential_growth_d(g, self.tileheight)

    """
    Stores the real part of Omega a gpuarray in constant memory.
    """
    def cache_omega_real(self, X):
        X = np.require(X, dtype = np.double, requirements=['A','W','O','C'])
        Xd = gpuarray.to_gpu(X)
        cuda.memcpy_dtod(self.Xd, Xd.ptr, Xd.nbytes)

    """
    Stores the imaginary part of Omega, and the cholesky decomposition of the imaginary part
    of omega as gpuarrays in constant memory
    """
    def cache_omega_imag(self, Yinv, T):
        Yinv = np.require(Yinv, dtype = np.double, requirements=['A','W','O','C'])
        T = np.require(T, dtype = np.double, requirements=['A','W','O','C'])
        Yinvd = gpuarray.to_gpu(Yinv)
        Td = gpuarray.to_gpu(T)
        cuda.memcpy_dtod(self.Td, Td.ptr, Td.nbytes)
        cuda.memcpy_dtod(self.Yinv_vd, Yinvd.ptr, Yinvd.nbytes)
        cuda.memcpy_dtod(self.Yinv_ud, Yinvd.ptr, Yinvd.nbytes)

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
    Computes the oscillatory part of the riemann-theta function without any derivatives
    across many different points. Z is the set of points to compute across.
    """
    def compute_v_without_derivs(self, Z):
        #Turn the numpy set Z into gpuarrays
        x = Z.real
        y = Z.imag
        x = np.require(x, dtype = np.double, requirements=['A','W','O','C'])
        y = np.require(y, dtype = np.double, requirements=['A','W','O','C'])
        xd = gpuarray.to_gpu(x)
        yd = gpuarray.to_gpu(y)
        self.yd = yd
        #Detemine N = the number of integer points to sum over and
        #         K = the number of values to compute the function at
        N = self.Sd.size/self.g
        K = Z.size/self.g
        #Create room on the gpu for the real and imaginary finite sum calculations
        fsum_reald = gpuarray.zeros(N*K, dtype=np.double)
        fsum_imagd = gpuarray.zeros(N*K, dtype=np.double)
        #Make all scalars into numpy data types
        Nd = np.int32(N)
        Kd = np.int32(K)
        gd = np.int32(self.g)
        blocksize = (self.tilewidth, self.tileheight, 1)
        gridsize = (N//self.tilewidth + 1, K//self.tileheight + 1, 1)
        self.finite_sum_without_derivs(fsum_reald, fsum_imagd, xd, yd, 
                     self.Sd, gd, Nd, Kd,
                     block = blocksize,
                     grid = gridsize)
        cuda.Context.synchronize()
        fsums_real = self.sum_reduction(fsum_reald, N, K, Kd, Nd)
        fsums_imag = self.sum_reduction(fsum_imagd, N, K, Kd, Nd)
        return fsums_real + 1.0j*fsums_imag

    """
    Computes the oscillatory part of the riemann-theta function with derivatives
    across many different points. Z is the set of points to compute across.
    """
    def compute_v_with_derivs(self, Z, derivs):
        #Turn the numpy set Z into gpuarrays
        x = Z.real
        y = Z.imag
        x = np.require(x, dtype = np.double, requirements=['A','W','O','C'])
        y = np.require(y, dtype = np.double, requirements=['A','W','O','C'])
        xd = gpuarray.to_gpu(x)
        yd = gpuarray.to_gpu(y)
        self.yd = yd
        #Turn the set of derivatives into gpuarray
        deriv_real = np.require(derivs.real, dtype = np.double, requirements=['A','W','O', 'C'])
        deriv_imag = np.require(derivs.imag, dtype = np.double, requirements=['A', 'W', 'O', 'C'])
        nderivs = len(derivs)
        deriv_reald = gpuarray.to_gpu(deriv_real)
        deriv_imagd = gpuarray.to_gpu(deriv_imag)
        nderivs_d = np.int32(nderivs)
        #Detemine N = the number of integer points to sum over and
        #         K = the number of values to compute the function at
        N = self.Sd.size/self.g
        K = Z.size/self.g
        #Create room on the gpu for the real and imaginary finite sum calculations
        fsum_reald = gpuarray.zeros(N*K, dtype=np.double)
        fsum_imagd = gpuarray.zeros(N*K, dtype=np.double)
        #Make all scalars into numpy data types
        Nd = np.int32(N)
        Kd = np.int32(K)
        gd = np.int32(self.g)
        blocksize = (self.tilewidth, self.tileheight, 1)
        gridsize = (N//self.tilewidth + 1, K//self.tileheight + 1, 1)
        self.finite_sum_with_derivs(fsum_reald, fsum_imagd, xd, yd, 
                     self.Sd, deriv_reald, deriv_imagd, nderivs_d, gd, Nd, Kd,
                     block = blocksize,
                     grid = gridsize)
        cuda.Context.synchronize()
        fsums_real = self.sum_reduction(fsum_reald, N, K, Kd, Nd)
        fsums_imag = self.sum_reduction(fsum_imagd, N, K, Kd, Nd)
        return fsums_real + 1.0j*fsums_imag


    #Perform parallel sum reduction on the computational grid consisting of 
    #all the partial sums of all the values of Z. The function returns approximations
    #of the real and imaginary parts of the riemann theta function for every point in
    #Z
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

    def compute_u(self):
        yd = self.yd
        y_len = len(yd)
        ud = gpuarray.zeros(y_len, dtype = np.double)
        blocksize = (self.g, self.tileheight, 1)
        gridsize = (1, (y_len - 1)//self.tileheight + 1,1)
        self.dot_prod(yd, ud, np.int32(self.g), np.int32(y_len), 
               block = blocksize, grid = gridsize)
        cuda.Context.synchronize()
        u = ud.get()
        return u




    """
    CUDA FUNCTIONS
    
    (fun1) contains function for computing the oscillatory part of the riemanntheta
    function
    
    (grid_sum_reduction_d) Is a function for computing sum reduction across the rows of a matrix
    
    (exponential_growth_d) Is a function for computing the exponential part of the riemanntheta function
    """
        
    def finite_sum_d(self, TILEWIDTH, TILEHEIGHT, g):
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
    __device__ void deriv_prod(int g, double* Sd_s, double* yd_s, double* dpr, double* dpi,
                                 double* deriv_real, double* deriv_imag, int nderivs)
    {
      int tx = threadIdx.x;
      int ty = threadIdx.y;

      double total_real = 1;
      double total_imag = 0;

      int i,j,k;
      for (i = 0; i < nderivs; i++){
        double term_real = 0;
        double term_imag = 0;
        for (j = 0; j < g; j++){
          double shift_j = 0;
          for (k = 0; k < g; k++){
            shift_j += Yinvd[j*g + k] * yd_s[ty*g + k];
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
        double dpr[1];
        double dpi[1];
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
        return (func, deriv_func, Xd, Yinvd, Td)

    def grid_sum_reduction_d(self):
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

    def exponential_growth_d(self, g, TILEHEIGHT):
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
