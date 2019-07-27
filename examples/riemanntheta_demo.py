"""
Grady Williams
January 28, 2013

This module provides functions for displaying graphs of the Riemann-Theta
function. There are 12 different graphs that can be generated, 10 of them
correspond to the graphics shown on the Digital Library of Mathematical 
Functions page for Riemann Theta (dlmf.nist.gov/21.4) and the names of the
functions that generate those plots correspond to the names of the plots on 
that page. (e.g plt_a1 plots generates the plot denoted a1 on the dlmf page).
The other two graphs are of the first and second derivatives for a given Omega.

Besides the plots for derivatives all of the plots have a few optional commands:

SIZE: Is the number of grid-points per direction over which the function is computed over, the 
default is set to 75.

warp: Is the mayavi warp number documentation for it can be found at: 
(docs.enthough.com/mayavi/mayavi/auto/mlab_helper_functions.html). The default is auto.

d_axes: Is a boolean value which determines whether or not the axis are displayed.

WARNING: If d_axis is set to True, be then warp should be set to '1'. Otherwise incorrect
axis will be displayed and function values will appear incorrect.

There are 3 different Omegas that are considered

Omega 1 = [[1.690983006 + .951056516*1.0j   1.5 + .363271264*1.0j]
          [1.5 + .363271264*1.0j            1.309016994 + .951056516*1.0j]]

Omega 2 = [[1.0j  -.5]
           [-.5   1.0j]]

Omega 3 = [[-.5 + 1.0j     .5 -.5*1.0j  -.5-.5*1.0j]
           [.5 -.5*1.0j    1.0j         0          ]
           [-.5 - .5*1.0j  0            1.0j       ]]

In all of the following graphs, the exponential growth of Riemann Theta has been factored out. 
"""
from abelfunctions import RiemannTheta
import numpy as np
from mayavi.mlab import *
import matplotlib.pyplot as plt

gpu = True
try:
    import pycuda.driver
except ImportError:
    gpu = False

"""
Plots the real part of Riemann Theta for Omega 1 with z = (x + iy,0)
where x,y are real numbers such that 0 < x < 1, 0 < y < 5  
corresponds to 21.4.1.a1 on DLMF
""" 
def plt_a1(SIZE=75, warp="auto", d_axes=False):
    X,Y,V = get_r1_vals(SIZE, gpu)
    V = V.real
    s = surf(X,Y,V, warp_scale = warp)
    if d_axes:
        axes()
    return s

"""
Plots the imaginary part of Riemann Theta for Omega 1 with z = (x + iy,0)
where x,y are real numbers such that 0 < x < 1, 0 < y < 5  
corresponds to 21.4.1.b1 on DLMF
""" 
def plt_b1(SIZE=75,warp="auto", d_axes=False):
    X,Y,V = get_r1_vals(SIZE,gpu)
    V = V.imag
    s = surf(X,Y,V, warp_scale = warp)
    if d_axes:
        axes()
    return s

"""
Plots the modulus of Riemann Theta for Omega 1 with z = (x + iy,0)
where x,y are real numbers such that 0 < x < 1, 0 < y < 5  
corresponds to 21.4.1.c1 on DLMF
""" 
def plt_c1(SIZE=75, warp="auto", d_axes=False):
    X,Y,V = get_r1_vals(SIZE, gpu)
    V = np.absolute(V)
    s = surf(X,Y,V, warp_scale = warp)
    if d_axes:
        axes()
    return s

def plt_a2(SIZE=75,warp = "auto",d_axes=False):
    X,Y,V = get_r2_vals(SIZE, gpu)
    V = V.real
    s = surf(X,Y,V, warp_scale = warp)
    if d_axes:
        axes()
    return s

def plt_b2(SIZE=75,warp= "auto", d_axes=False):
    X,Y,V = get_r2_vals(SIZE, gpu)
    V = V.imag
    s = surf(X,Y,V, warp_scale = warp)
    if d_axes:
        axes()
    return s

def plt_c2(SIZE=75, warp = "auto", d_axes=False):
    X,Y,V = get_r2_vals(SIZE, gpu)
    V = np.absolute(V)
    s = surf(X,Y,V,warp_scale = warp)
    if d_axes:
        axes()
    return s

def plt_a3(SIZE=75, warp = "auto", d_axes=False):
    X,Y,V = get_r3_vals(SIZE, gpu)
    V = V.real
    s = surf(X,Y,V, warp_scale = warp)
    if d_axes:
        axes()
    return s

def plt_b3(SIZE=75, warp= "auto", d_axes=False):
    X,Y,V = get_r3_vals(SIZE,gpu)
    V = V.imag
    s = surf(X,Y,V, warp_scale = warp)
    if d_axes:
        axes()
    return s

def plt_c3(SIZE=75, warp= "auto", d_axes=False):
    X,Y,V = get_r3_vals(SIZE,gpu)
    V = np.absolute(V)
    s = surf(X,Y,V,warp_scale = warp)
    if d_axes:
        axes()
    return s

def plt_21_4_2(SIZE=75, warp = "auto", d_axes = False):
    X,Y,V = get_d_vals(SIZE, gpu)
    V = V.real
    s = surf(X,Y,V, warp_scale = warp)
    if d_axes:
        axes()
    return s

def plt_21_4_3(SIZE=75, warp = "auto", d_axes=False):
    theta = RiemannTheta
    Omega = np.matrix([[1.0j, -.5], [-.5,1.0j]])
    X,Y = np.mgrid[0:1:SIZE*1.0j, 0:2:SIZE*1.0j]
    Z = X + Y*1.0j
    Z = Z.flatten()
    U,V = theta.exp_and_osc_at_point([[z,0] for z in Z], Omega, batch=True)
    V = np.absolute(V)
    V = V.reshape(SIZE,SIZE)
    s = surf(X,Y,V, warp_scale = warp)
    if d_axes:
        axes()
    return s

def plt_21_4_4(SIZE=75,warp = "auto", d_axes=False, gpu=False):
    theta = RiemannTheta
    Omega = np.matrix([[1.0j, -.5], [-.5,1.0j]])
    X,Y = np.mgrid[0:4:SIZE*1.0j, 0:4:SIZE*1.0j]
    Z = X + Y*1.0j
    Z = Z.flatten()
    U,V = theta.exp_and_osc_at_point([[z.real*1.0j,z.imag*1.0j] for z in Z], Omega, batch=True)
    V = V.real
    V = V.reshape(SIZE,SIZE)
    s = surf(X,Y,V, warp_scale = warp)
    if d_axes:
        axes()
    return s

def plt_21_4_5(SIZE=75,warp = "auto", d_axes=False, gpu=False):
    theta = RiemannTheta
    Omega = np.matrix([[-.5 + 1.0j, .5 -.5*1.0j, -.5-.5*1.0j],
                       [.5 -.5*1.0j, 1.0j, 0],
                       [-.5 - .5*1.0j, 0, 1.0j]])
    X,Y = np.mgrid[0:1:SIZE*1.0j, 0:3:1.0j*SIZE]
    Z = X+Y*1.0j
    Z = Z.flatten()
    U,V = theta.exp_and_osc_at_point([[z,0,0] for z in Z], Omega, batch=True)
    V = V.real
    V = V.reshape(SIZE,SIZE)
    s = surf(X,Y,V,warp_scale=warp)
    if d_axes:
        axes()
    return s

def plt_first_deriv():
    theta = RiemannTheta
    Omega = np.matrix([[1.690983006 + .951056516*1.0j, 1.5 + .363271264*1.0j],
                       [1.5 + .363271264*1.0j, 1.309016994 + .951056516*1.0j]])
    k = [[1,0]]
    Z = np.linspace(0,50,500)
    U,V = theta.exp_and_osc_at_point([[0, z*1.0j] for z in Z], Omega, deriv=k, batch=True)
    plt.plot(Z, V.real)
    plt.show()

def plt_second_deriv():
    theta = RiemannTheta
    Omega = np.matrix([[1.690983006 + .951056516*1.0j, 1.5 + .363271264*1.0j],
                       [1.5 + .363271264*1.0j, 1.309016994 + .951056516*1.0j]])
    k = [[1,0],[1,0]]
    Z = np.linspace(0,50,500)
    U,V = theta.exp_and_osc_at_point([[0, z*1.0j] for z in Z], Omega, deriv=k, batch=True)
    plt.plot(Z, V.real)
    plt.show()

def explosion(SIZE, gpu):
    theta = RiemannTheta
    Omega = np.matrix([[1.690983006 + .951056516*1.0j, 1.5 + .363271264*1.0j],
                       [1.5 + .363271264*1.0j, 1.309016994 + .951056516*1.0j]])
    X,Y = np.mgrid[-1.5:1.5:SIZE*1.0j, -1.5:1.5:SIZE*1.0j]
    Z = X + Y*1.0j
    Z = Z.flatten()
    U,V = theta.exp_and_osc_at_point([[z,0] for z in Z], Omega, batch=True)
    V = np.exp(U)*V
    V = V.reshape(SIZE, SIZE)
    s = surf(X,Y,np.absolute(V), warp_scale = 'auto')
    savefig("test.eps")
    
def get_r1_vals(SIZE, gpu):
    theta = RiemannTheta
    Omega = np.matrix([[1.690983006 + .951056516*1.0j, 1.5 + .363271264*1.0j],
                       [1.5 + .363271264*1.0j, 1.309016994 + .951056516*1.0j]])
    X,Y = np.mgrid[0:1:SIZE*1.0j, 0:5:SIZE*1.0j]
    Z = X + Y*1.0j
    Z = Z.flatten()
    U,V = theta.exp_and_osc_at_point([[z,0] for z in Z], Omega, batch=True)
    V = V.reshape(SIZE, SIZE)
    return X,Y,V

def get_r2_vals(SIZE, gpu):
    theta = RiemannTheta
    Omega = np.matrix([[1.690983006 + .951056516*1.0j, 1.5 + .363271264*1.0j],
                      [1.5 + .363271264*1.0j, 1.309016994 + .951056516*1.0j]])
    X = np.linspace(0,1,SIZE)
    Y = np.linspace(0,1,SIZE)
    Z = []
    for x in X:
        for y in Y:
          Z.append([x,y])
    U,V = theta.exp_and_osc_at_point(Z, Omega, batch=True)
    V = V.reshape(SIZE,SIZE)
    return X,Y,V

def get_r3_vals(SIZE, gpu):
    theta = RiemannTheta
    Omega = np.matrix([[1.690983006 + .951056516*1.0j, 1.5 + .363271264*1.0j],
                       [1.5 + .363271264*1.0j, 1.309016994 + .951056516*1.0j]])
    X,Y = np.mgrid[0:5:SIZE*1.0j, 0:5:SIZE*1.0j]
    Z = X + Y*1.0j
    Z = Z.flatten()
    U,V = theta.exp_and_osc_at_point([[1.0j*z.real,1.0j*z.imag] for z in Z], Omega, batch=True)
    V = V.reshape(SIZE, SIZE)
    return X,Y,V

def get_d_vals(SIZE, gpu):
    theta = RiemannTheta
    Omega = np.matrix([[1.0j, -.5], [-.5, 1.0j]])
    X,Y = np.mgrid[0:1:SIZE*1.0j, 0:5:SIZE*1.0j]
    Z = X + Y * 1.0j
    Z = Z.flatten()
    U,V = theta.exp_and_osc_at_point([[z,0] for z in Z], Omega, batch=True)
    V = V.reshape(SIZE,SIZE)
    return X,Y,V
