print "===== abelfunctions Demo Script ====="
from abelfunctions import *
from sympy.abc import x,y

#f = y**3 + 2*x**3*y - x**7
f = y**2 - (x-1)*(x+1)*(x-2)*(x+2)
X = RiemannSurface(f,x,y)

print "\n\tRS"
print X

print "\n\tRS: monodromy"
mon = X.monodromy()
print mon
X.show_paths()

print "\n\tRS: homology"
hom = X.homology()
for key,value in hom.iteritems():
    print key
    print value
    print

# print "\n\tc-cycle(0)"
# gamma = X.c_cycle(0)

print "\n\tRS: period matrix"
A,B = X.period_matrix()
print A
print B
print
print "abelfunctions: tau =", B[0][0]/A[0][0]
print "maple:         tau = 0.999999 + 1.563401 I"


