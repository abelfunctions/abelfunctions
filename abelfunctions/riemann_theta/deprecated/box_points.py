"""
Grady Williams
May 2013

This program computes all of the integer points inside an n-dimensional box
of radius R.
"""

import time
import numpy as np
import scipy.linalg as la
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
from riemanntheta_cuda import RiemannThetaCuda

def riemanntheta_high_dim(X, Yinv, T, z, g, rad, max_points = 10000000):
    parRiemann = RiemannThetaCuda(1,512)
    #initialize parRiemann
    parRiemann.compile(g)
    parRiemann.cache_omega_real(X)
    parRiemann.cache_omega_imag(Yinv,T)
    #compile the box_points program
    point_finder = func1()
    R = get_rad(T, rad)
    print R
    num_int_points = (2*R + 1)**g
    num_partitions = num_int_points//max_points
    num_final_partition = num_int_points - num_partitions*max_points
    osc_part = 0 + 0*1.j
    if (num_partitions > 0):
        S = gpuarray.zeros(np.int(max_points * g), dtype=np.double)
    print "Required number of iterations"
    print num_partitions
    print 
    for p in range(num_partitions):
        print p
        print
        S = box_points(point_finder, max_points*p, max_points*(p+1),g,R, S)
        parRiemann.cache_intpoints(S, gpu_already=True)
        osc_part += parRiemann.compute_v_without_derivs(np.array([z]))
    S = gpuarray.zeros(np.int((num_int_points - num_partitions*max_points)*g), dtype = np.double)
    print num_partitions*max_points,num_int_points
    S = box_points(point_finder, num_partitions*max_points, num_int_points, g, R,S)
    parRiemann.cache_intpoints(S,gpu_already = True)
    osc_part += parRiemann.compute_v_without_derivs(np.array([z]))
    print osc_part
    return osc_part

def get_rad(T, rad):
    ell = np.dot(T, T.transpose())*rad
    eigs = la.eigvals(ell).real
    eigs = np.absolute(eigs)
    R = np.ceil(max(eigs))
    return np.int64(R)
    
def box_points(point_finder, min_p,max_p,g,r, S):
    num_points = max_p - min_p
    min_p_d = gpuarray.to_gpu(np.array([min_p], dtype = np.int64))
    max_p_d = gpuarray.to_gpu(np.array([max_p], dtype = np.int64))
    g_d = gpuarray.to_gpu(np.array([g], dtype = np.int64))
    r_d = gpuarray.to_gpu(np.array([r], dtype = np.int64))
    blocksize = (10, 1, 1)
    gridsize = (1000, 1000, 1)
    cuda.Context.synchronize()
    point_finder(S, min_p_d,max_p_d, g_d, r_d, block=blocksize, grid=gridsize)
    cuda.Context.synchronize()
    return S

def checklist(S):
    B = S.reshape(len(S)/2,2)
    print B[10000000:]
    
def func1():
    template = """

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

__global__ void point_finder(double* S, int* start, int* max_p, int* g, int* r)
{
   int tdx = threadIdx.x;
   int bdx = blockIdx.x;
   int bdy = blockIdx.y;
   int index = 1000*(10)*bdy + 10*bdx + tdx + start[0];
   if (index < max_p[0]) {
     int i = 0;
     for (i = 0; i < g[0]; i++) {
       int p = 1;
       int j = 0;
       for (j = 0; j < i; j++){
         p *= 2*r[0] + 1;
       }
       S[g[0]*(index - start[0]) + i] = ((index/p) % (2*r[0] + 1)) - r[0];
     }
   }
}
"""
    mod = SourceModule(template)
    func = mod.get_function("point_finder")
    return func

    
if __name__=="__main__":
    print "Testing GPU capacity"
    rad = 100
    for i in range(1,32):
        box_points(10**i, 0)
        print i
