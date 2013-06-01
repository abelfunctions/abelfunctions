"""
Grady Williams

Performs Siegel reduction for a given Riemann Matrix
"""
import numpy as np
from lattice_reduction import *
import scipy.linalg as la

def siegel(Om, g):
    Omega = np.copy(Om)
    #Initialize gamma, a,b,c, and d
    gamma = np.identity(2*g, dtype = np.double)
    a = np.zeros((g,g), dtype = np.double)
    for i in range(g-1):
        a[g-1-i,g-1-i] = 1
    d = np.copy(a)
    b = np.zeros((g,g), dtype = np.double)
    b[0,0] = -1
    c = np.zeros((g,g), dtype = np.double)
    c[0,0] = 1
    transformed = False
    count = 0
    while (not transformed):
        count += 1
        Y = Omega.imag
        T = la.cholesky(Y)
        #Perform lattice reduction on T
        L = lattice_reduce(T)
        #Sort L by the norm of the columns of T
        L = normSort(L,g)
        #Solve T*U = L
        U = np.dot(la.inv(T), L)
        #Get the product U.Transpose * Omega * U
        Omega = np.dot(Omega, U)
        Omega = np.dot(U.transpose(), Omega)
        #Adjust gamma
        Uinv = la.inv(U)
        mod = np.zeros((2*g, 2*g),dtype = np.double)
        mod[0:g, 0:g] = U.transpose()
        mod[g:, g:] = Uinv
        gamma = np.dot(mod,gamma)
        #Omega = Omega - Re(Omega)
        _X_ = np.rint(Omega.real)
        Omega = Omega - _X_
        mod = np.identity(2*g, np.double)
        mod[:g, g:] = -_X_
        gamma = np.dot(mod,gamma)
        if (Omega[0,0].real**2 + Omega[0,0].imag**2 < 1):
            mod[:g, :g] = a
            mod[:g, g:] = b
            mod[g:, :g] = c
            mod[g:, g:] = d
            gamma = np.dot(mod,gamma)
            #Compute (a*Omega + b)*(c*Omega + d)^(-1)
            H = np.dot(a,Omega) + b
            K = np.dot(c,Omega) + d
            K = la.inv(K)
            Omega = np.dot(H,K)
        else:
            transformed = True
    return Omega, gamma

#Performs the insertion sort algorithm on the columns of a matrix
def normSort(L,g):
    column_vals = []
    for i in range(g):
        column_vals.append((i,la.norm(L[:,i])))
    for i in range(1, g):
        in_order = False
        j = i
        while (not in_order and j > 0):
            if (column_vals[j][1] < column_vals[j-1][1]):
                temp = column_vals[j]
                column_vals[j] = column_vals[j-1]
                column_vals[j-1] = temp
                j -= 1
            else:
                in_order = True
    b = np.zeros((g,g))
    for i in range(g):
        k = column_vals[i][0]
        b[:, i] = L[:, k]
    return b

def test():
    Omega = -1.0/(2 * np.pi * 1.0j) * np.array([[111.207, 96.616], [96.616, 83.943]])
    g = 2
    om, gam = siegel(Omega, 2)
    print "New Omega: "
    print om
    print
    print "Gamma:"
    print gam
                
            
if __name__=="__main__":
    test()
