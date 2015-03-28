import numpy as np
from riemanntheta import RiemannTheta_Function
import pylab as p
import matplotlib.pyplot as plt
from pycuda import gpuarray
import time

def demo1():
    theta = RiemannTheta_Function()
    Omega = np.array([[1.j, .5], [.5, 1.j]])
    print
    print "Calculating 3,600 points of the Riemann Theta Function in C..."
    print
    print "Omega = [i   .5]"
    print "        [.5   i]"
    print 
    print "z = (x + iy, 0) where (0 < x < 1) and (0 < y < 5)"
    SIZE = 60
    x = np.linspace(0,1,SIZE)
    y = np.linspace(0,5,SIZE)
    X,Y = p.meshgrid(x,y)
    Z = X + Y*1.0j
    Z = Z.flatten()
    start = time.clock()
    U,V = theta.exp_and_osc_at_point([[z,0] for z in Z], Omega, gpu=False, batch=True)
    done = time.clock() - start
    print "Time to perform the calculation: " + str(done)
    print
    Z = (V.reshape(60,60)).imag
    print "\Plotting the imaginary part of the function..."
    plt.contourf(X,Y,Z,7,antialiased=True)
    plt.show()

def demo2():
    theta = RiemannTheta_Function()
    Omega = np.array([[1.j, .5], [.5, 1.j]])
    print
    print "Calculating 3,600 points of the Riemann Theta Function on GPU..."
    print
    print "Omega = [i   .5]"
    print "        [.5   i]"
    print 
    print "z = (x + iy, 0) where (0 < x < 1) and (0 < y < 5)"
    SIZE = 60
    x = np.linspace(0,1,SIZE)
    y = np.linspace(0,5,SIZE)
    X,Y = p.meshgrid(x,y)
    Z = X + Y*1.0j
    Z = Z.flatten()
    start = time.clock()
    U,V = theta.exp_and_osc_at_point([[z,0] for z in Z], Omega, batch=True)
    done = time.clock() - start
    print "Time to perform the calculation: " + str(done)
    print
    Z = (V.reshape(60,60)).imag
    print "\Plotting the imaginary part of the function..."
    plt.contourf(X,Y,Z,7,antialiased=True)
    plt.show()

def demo3():
    theta = RiemannTheta_Function()
    Omega = np.array([[1.j, .5], [.5, 1.j]])
    print
    print "Calculating 1,000,000 points of the Riemann Theta Function on GPU..."
    print
    print "Omega = [i   .5]"
    print "        [.5   i]"
    print 
    print "z = (x + iy, 0) where (0 < x < 1) and (0 < y < 5)"
    SIZE = 1000
    x = np.linspace(0,1,SIZE)
    y = np.linspace(0,5,SIZE)
    X,Y = p.meshgrid(x,y)
    Z = X + Y*1.0j
    Z = Z.flatten()
    Z = [[z,0] for z in Z]
    print "Starting computation on the GPU"
    start = time.clock()
    U,V = theta.exp_and_osc_at_point(Z, Omega, batch=True)
    done = time.clock() - start
    print "Time to perform the calculation: " + str(done)
    print
    print "Starting computation on the CPU"
    start = time.clock()
    U,V = theta.exp_and_osc_at_point(Z, Omega, batch=True, gpu=False)
    done = time.clock() - start
    print "Time to perform the calculation: " + str(done)
    print


def demo4():
    theta = RiemannTheta_Function()
    z = np.array([1.j, .5*1.j, 1.j])
    omegas = []
    I = 1.j
    t_vals = np.linspace(1, 0, 10000)
    for t in t_vals:
        a = np.array([[-0.5*t + I, 0.5*t*(1-I), -0.5*t*(1 + I)],
                      [0.5*t*(1-I), I, 0],
                      [-0.5*t*(1+I), 0, I]])
        omegas.append(a)
    print "z = (i, i, i), calculating z for 10,000 different Omegas"
    print
    print "Omegas  =  [-0.5*(1-t) + i    0.5*t*(1-i)    -0.5*(1-t)*(1 + i)]"
    print "           [0.5*(1-t)*(1-i)             i                     0]"
    print "           [-0.5*(1-t)*(1+i)            0                     i]"
    print "for 0 <= t <= 1"
    print
    print "Beginning Computation on the GPU"
    start = time.clock()
    v = theta.multiple_omega_process1(z, omegas, 3)
    print "Computation Finished, elapsed time: " + str(time.clock() - start)
    print
    print v
    print
    print "================================="
    print "Beginning Computation on the CPU"
    start = time.clock()
    u = []
    for i in range(10000):
        U,V = theta.exp_and_osc_at_point(z,omegas[i])
        u.append(V)
    print "Computation Finished, elapsed time: " + str(time.clock() - start)
    print
    print np.array(v)


