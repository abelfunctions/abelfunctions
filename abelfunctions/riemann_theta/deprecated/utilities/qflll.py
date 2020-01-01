import sys
import numpy as np
import ctypes
import ctypes.util

# try to import the PARI library
try:
    pari = ctypes.cdll.LoadLibrary(ctypes.util.find_library('pari'))
    pari.pari_init(4000000,2)
    pari.gp_read_str.restype = ctypes.POINTER(ctypes.c_int)
    pari.qflll0.restype = ctypes.POINTER(ctypes.c_int)
    pari.GENtostr.restype = ctypes.POINTER(ctypes.c_char)
except OSError:
    print "Could not import PARI/GP library. Please install and try again."
    sys.exit(0)



def qflll(mat):
    # convert the matrix to an appropriately formatted string
    M = mat.shape[0]
    p = [',',';']
    elts = ''.join([str(mat.item(i))+p[0 if (i+1)%M else 1] 
                    for i in range(M**2)])[:-1] 
    
    # compute lll reduction and read in PARI/GP's resulting string
    gen = pari.gp_read_str('[' + elts + ']')
    gen = pari.qflll0(gen,ctypes.c_long(0))
    s = pari.GENtostr(gen)
    s = ctypes.string_at(s)

    # convert string into numpy matrix and return
    mat = s.replace('\n','').replace('][',';')[:-1] + ']'
    mat = np.matrix(mat)
    return mat

