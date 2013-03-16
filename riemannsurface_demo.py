print "===== abelfunctions Demo Script ====="
from abelfunctions import *
from sympy.abc import x,y

#f = y**3 + 2*x**3*y - x**7
f = y**2 - (x-1)*(x+1)*(x-2)*(x+2)
X = RiemannSurface(f,x,y)

print "\n\tRS"
print X

print "\n\tRS: monodromy"
base_point, base_sheets, branch_points, mon, G = X.monodromy()
print "base_point:"
print base_point
print "base_sheets:"
for s in base_sheets: print s
print "branch points:"
for b in branch_points: print b
print "monodromy group:"
for m in mon: print m

X.show_paths()

print "\n\tRS: homology"
hom = X.homology()
print "genus:"
print hom['genus']
print "cycles:"
for c in hom['cycles']: print c
print "lincomb:"
print hom['linearcombination']

print "\n\tRS: computing cycles"
gamma = [X.c_cycle(i) for i in xrange(len(hom['cycles']))]

# print "\n\tRS: period matrix"
# A,B = X.period_matrix()
# print A
# print B
# print
# print "abelfunctions: tau =", B[0][0]/A[0][0]
# print "maple:         tau = 0.999999 + 1.563401 I"


