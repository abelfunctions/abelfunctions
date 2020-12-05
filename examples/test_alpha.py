"""
Test script for performing certified analytic continuation
"""

import pdb
import abelfunctions as ab
import numpy
import sympy
import matplotlib.pyplot as plt

from abelfunctions.monodromy import monodromy_graph, show_paths
from abelfunctions.riemannsurface_path import path_around_branch_point




def coeff_functions(p,x):
    return map(lambda ak: sympy.lambdify(x,ak,'numpy'), p.all_coeffs())
    
    
def mu(f,a,n,x,y):
    binom = sympy.binomial_coefficients_list(n)
    binom = map(numpy.double, binom)
    normf = numpy.sqrt(sum(numpy.abs(a[k](x))**2 * 1/binom[k] 
                           for k in range(n+1)
                           )
                       )
    Delta = numpy.sqrt(numpy.double(n)) * (1 + numpy.abs(y))
    normy = 1 + numpy.abs(y)

    return normf * Delta / (2 * normy)
    
    


def analytically_continue(df,a,n,fibre,t,path):
    alpha0 = (13.0 - 2.0*numpy.sqrt(17.0))/4.0
    t0 = 0

    maxdt = (t-t0)/16.0

    ti = t0
    dt = maxdt
    xi = path(t0)
    yi = [yj for yj in fibre]
    ypath = [fibre]

    while ti < t:
        # iterate and approximate
        print("=== Iterate ===")
        print("\t fibre  =", fibre)
        tip1 = ti + dt
        xip1 = path(tip1)
        yapp = yi

        # we have to check quadratic basin stuff for _every_ root
        # simultaneously or else we might have a branch jump
        are_approximate_solutions = True                
        beta_yapp = [beta(df,xip1,yappj) for yappj in yapp]
        gamma_yapp = [gamma(df,n,xip1,yappj) for yappj in yapp]
        alpha_yapp = [beta_yapp[j] * gamma_yapp[j] for j in range(n)]
        print("\t alphas = %s (alpha0 = %f)" % (alpha_yapp, alpha0))
        for j in range(n):
            # check if our guess is in the quadratic convergence basin
            alphaj = beta_yapp[j] * gamma_yapp[j]
            if alphaj >= alpha0:
                are_approximate_solutions = False
                break

        print("\t approximate solns? %s" % (are_approximate_solutions))

        # only perform separated basins detection if the roots are
        # close enough
        separated_basins = True
        if are_approximate_solutions:
            # now that we know the solutions are approximate solutions, we
            # use the basin radius formula to ensure that each approximate
            # root is within only one quadradic convergence basin.
            beta_yapp = [beta(df, xip1, yappj) for yappj in yapp]
            for j in range(n):
                betaj = beta_yapp[j]
                for k in range(n):
                    if j != k:
                        dist = numpy.abs(yapp[j] - yapp[k])
                        betak = beta_yapp[k]

                        # use triangle inequality to guarantee separation
                        # since we don't know what the actual roots are
                        if dist <= 2*(betaj + betak):
                            print("\t === TEST: dist <= 2(betak + betak) ===")
                            print("\t betaj =", betaj)
                            print("\t betak =", betak)
                            print("\t bound =", 2*(betaj+betak))
                            print("\t dist  =", dist)
                            separated_basins = False
                            break

        print("\t separated_basins?  %s" % (separated_basins))
        print("\t t  = %s" % (ti))
        print("\t dt = %s" % (dt))
        if dt < 1e-6: break

        if are_approximate_solutions and separated_basins:
            newy = [newton(df[0],df[1],xip1,yappj) for yappj in yapp]
            ypath.append(newy)
            ti = tip1
            xi = xip1
            yi = newy
            dt = min(2*dt,maxdt)
        else:
            dt *= 0.5

    return ypath
    

if __name__=='__main__':
    from sympy.abc import x,y

    # compute f and its derivatives with respect to y along with
    # additional data
    f1 = (x**2 - x + 1)*y**2 - 2*x**2*y + x**4
    f2 = -x**7 + 2*x**3*y + y**3
    f3 = (y**2-x**2)*(x-1)*(2*x-3) - 4*(x**2+y**2-2*x)**2
    f4 = y**2 + x**3 - x**2
    f5 = (x**2 + y**2)**3 + 3*x**2*y - y**3
    f6 = y**4 - y**2*x + x**2   # case with only one finite disc pt
    f7 = y**3 - (x**3 + y)**2 + 1
    f8 = (x**6)*y**3 + 2*x**3*y - 1
    f9 = 2*x**7*y + 2*x**7 + y**3 + 3*y**2 + 3*y
    f10= (x**3)*y**4 + 4*x**2*y**2 + 2*x**3*y - 1

    f = f9
    n = sympy.degree(f,y)
    coeffs = coeff_functions(sympy.poly(f,y),x)

    # compute the path parameterization around a given branch point
    bpt = 3
    G = monodromy_graph(f,x,y)
    path_segments = path_around_branch_point(G,bpt,1)
    nseg = len(path_segments)


    # select a root at the base_point
    a = G.nodes[0]['basepoint']
    fibre = map(numpy.complex,sympy.nroots(f.subs(x,a),n=15))

    # analytically continue
    ypath = []
    for k in range(nseg):
#        print("=== segment %k ===")
#        print("fibre (start) =")
#        for yj in fibre: print(yj)

        ypath += analytically_continue(df,coeffs,n,fibre,1,path_segments[k][0])
        fibre = [yij for yij in ypath[-1]]

#        print("fibre (end)   =")
#        for yj in fibre: print(yj)

    # parse out yroots data: right now it's of the form
    # 
    #     iter 0       item 1    ...
    #  (y1,...,yn), (y1,...,yn), ...
    #
    # but we want instead
    #
    #  (y1_0, y1_1, y1_2, ...)
    #  ...
    #  (yn_0, yn_1, yn_2, ...)
    #
    ys = zip(*ypath)

    # plot
    fig = plt.figure()
    for j in range(n):
        ypath = numpy.array(ys[j], dtype=numpy.complex)

        ax = fig.add_subplot(1,n,j+1)
        ax.plot(ypath.real, ypath.imag, 'g')
        ax.plot(ypath.real, ypath.imag, 'b.')
        ax.plot(ypath[0].real, ypath[0].imag, 'ks', markersize=12)
        ax.plot(ypath[-1].real, ypath[-1].imag, 'ro', markersize=10)

    fig.show()
