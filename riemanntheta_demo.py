"""
Grady Williams
January 28, 2013

This module provides functions for displaying graphs of the Riemann-Theta
function. 
"""

from abelfunctions import RiemannTheta
import numpy as np
from mayavi.mlab import *

def get_a_vals(SIZE, gpu):
    theta = RiemannTheta
    Omega = np.matrix([[1.690983006 + .951056516*1.0j, 1.5 + .363271264*1.0j],
                       [1.5 + .363271264*1.0j, 1.309016994 + .951056516*1.0j]])
    X,Y = np.mgrid[0:1:SIZE*1.0j, 0:5:SIZE*1.0j]
    Z = X + Y*1.0j
    Z = Z.flatten()
    U,V = theta.exp_and_osc_at_point([[z,0] for z in Z], Omega, List=True, gpu=gpu)
    V = V.reshape(SIZE, SIZE)
    return X,Y,V

def a_plt1(SIZE=75, warp="auto", d_axes=False, gpu=False):
    X,Y,V = get_a_vals(SIZE, gpu)
    V = V.real
    s = surf(X,Y,V, warp_scale = warp)
    if d_axes:
        axes()
    return s

def a_plt2(SIZE=75,warp="auto", d_axes=False, gpu=False):
    X,Y,V = get_a_vals(SIZE,gpu)
    V = V.imag
    s = surf(X,Y,V, warp_scale = warp)
    if d_axes:
        axes()
    return s

def a_plt3(SIZE=75, warp="auto", d_axes=False, gpu=False):
    X,Y,V = get_a_vals(SIZE, gpu)
    V = np.absolute(V)
    s = surf(X,Y,V, warp_scale = warp)
    if d_axes:
        axes()
    return s

def get_b_vals(SIZE, gpu):
    theta = RiemannTheta
    Omega = np.matrix([[1.690983006 + .951056516*1.0j, 1.5 + .363271264*1.0j],
                      [1.5 + .363271264*1.0j, 1.309016994 + .951056516*1.0j]])
    X = np.linspace(0,1,SIZE)
    Y = np.linspace(0,1,SIZE)
    Z = []
    for x in X:
        for y in Y:
          Z.append([x,y])
    U,V = theta.exp_and_osc_at_point(Z, Omega, List=True, gpu=gpu)
    V = V.reshape(SIZE,SIZE)
    return X,Y,V

def b_plt1(SIZE=75,warp = "auto",d_axes=False, gpu=False):
    X,Y,V = get_b_vals(SIZE, gpu)
    V = V.real
    s = surf(X,Y,V, warp_scale = warp)
    if d_axes:
        axes()
    return s

def b_plt2(SIZE=75,warp= "auto", d_axes=False, gpu=False):
    X,Y,V = get_b_vals(SIZE, gpu)
    V = V.imag
    s = surf(X,Y,V, warp_scale = warp)
    if d_axes:
        axes()
    return s

def b_plt3(SIZE=75, warp = "auto", d_axes=False, gpu=False):
    X,Y,V = get_b_vals(SIZE, gpu)
    V = np.absolute(V)
    s = surf(X,Y,V,warp_scale = warp)
    if d_axes:
        axes()
    return s

def get_c_vals(SIZE, gpu):
    theta = RiemannTheta
    Omega = np.matrix([[1.690983006 + .951056516*1.0j, 1.5 + .363271264*1.0j],
                       [1.5 + .363271264*1.0j, 1.309016994 + .951056516*1.0j]])
    X,Y = np.mgrid[0:5:SIZE*1.0j, 0:5:SIZE*1.0j]
    Z = X + Y*1.0j
    Z = Z.flatten()
    U,V = theta.exp_and_osc_at_point([[1.0j*z.real,1.0j*z.imag] for z in Z], Omega, List=True, gpu=gpu)
    V = V.reshape(SIZE, SIZE)
    return X,Y,V

def c_plt1(SIZE=75, warp = "auto", d_axes=False, gpu=False):
    X,Y,V = get_c_vals(SIZE, gpu)
    V = V.real
    s = surf(X,Y,V, warp_scale = warp)
    if d_axes:
        axes()
    return s

def c_plt2(SIZE=75, warp= "auto", d_axes=False, gpu=False):
    X,Y,V = get_c_vals(SIZE,gpu)
    V = V.imag
    s = surf(X,Y,V, warp_scale = warp)
    if d_axes:
        axes()
    return s

def c_plt3(SIZE=75, warp= "auto", d_axes=False, gpu=False):
    X,Y,V = get_c_vals(SIZE,gpu)
    V = np.absolute(V)
    s = surf(X,Y,V,warp_scale = warp)
    if d_axes:
        axes()
    return s

def get_d_vals(SIZE, gpu):
    theta = RiemannTheta
    Omega = np.matrix([[1.0j, -.5], [-.5, 1.0j]])
    X,Y = np.mgrid[0:1:SIZE*1.0j, 0:5:SIZE*1.0j]
    Z = X + Y * 1.0j
    Z = Z.flatten()
    U,V = theta.exp_and_osc_at_point([[z,0] for z in Z], Omega, List=True, gpu = gpu)
    V = V.reshape(SIZE,SIZE)
    return X,Y,V

def d_plt1(SIZE=75, warp = "auto", d_axes = False, gpu=False):
    X,Y,V = get_d_vals(SIZE, gpu)
    V = V.real
    s = surf(X,Y,V, warp_scale = warp)
    if d_axes:
        axes()
    return s

def d_plt2(SIZE=75, warp = "auto", d_axes=False, gpu=False):
    theta = RiemannTheta
    Omega = np.matrix([[1.0j, -.5], [-.5,1.0j]])
    X,Y = np.mgrid[0:1:SIZE*1.0j, 0:2:SIZE*1.0j]
    Z = X + Y*1.0j
    Z = Z.flatten()
    U,V = theta.exp_and_osc_at_point([[z,0] for z in Z], Omega, List=True, gpu=gpu)
    V = np.absolute(V)
    V = V.reshape(SIZE,SIZE)
    s = surf(X,Y,V, warp_scale = warp)
    if d_axes:
        axes()
    return s

def d_plt3(SIZE=75,warp = "auto", d_axes=False, gpu=False):
    theta = RiemannTheta
    Omega = np.matrix([[1.0j, -.5], [-.5,1.0j]])
    X,Y = np.mgrid[0:4:SIZE*1.0j, 0:4:SIZE*1.0j]
    Z = X + Y*1.0j
    Z = Z.flatten()
    U,V = theta.exp_and_osc_at_point([[z.real*1.0j,z.imag*1.0j] for z in Z], Omega, List=True, gpu=gpu)
    V = V.real
    V = V.reshape(SIZE,SIZE)
    s = surf(X,Y,V, warp_scale = warp)
    if d_axes:
        axes()
    return s

def e_plt1(SIZE=75,warp = "auto", d_axes=False, gpu=False):
    theta = RiemannTheta
    Omega = np.matrix([[-.5 + 1.0j, .5 -.5*1.0j, -.5-.5*1.0j],
                       [.5 -.5*1.0j, 1.0j, 0],
                       [-.5 - .5*1.0j, 0, 1.0j]])
    X,Y = np.mgrid[0:1:SIZE*1.0j, 0:3:1.0j*SIZE]
    Z = X+Y*1.0j
    Z = Z.flatten()
    U,V = theta.exp_and_osc_at_point([[z,0,0] for z in Z], Omega, List=True, gpu=gpu)
    V = V.real
    V = V.reshape(SIZE,SIZE)
    s = surf(X,Y,V,warp_scale=warp)
    if d_axes:
        axes()
    return s
