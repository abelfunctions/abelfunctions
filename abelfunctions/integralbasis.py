import sympy
import pdb 

from puiseux import puiseux
from sympy.abc import z


def valuation(p,x):
    """
    Given a collection of puiseux series, return the valuations. That is, the 
    exponents of the leading order term.
    """
    return p.expand(mult=True,force=True).leadterm(x)[1]

def Int(i,p):
    """
    The function .. math:

        Int_i = \sum_{k \neq i} v(p_i - p_k)
    """
    n   = len(p)
    pi  = p[i]
    val = 0
    for k in xrange(n):
        if k != i:
            val += valuation(pi-p[k],x)

    return val

def compute_expansion_bounds(p,x):
    """
    Computes the expansion bounds `N_1,\ldots,N_n` such that for all 
    polynomials `G \in L[x,y]` the truncation `r_i` of the Puiseux series
    `p_i` satisfying `v(r_i - p_i) > N_i` satisfies the relation .. math:

        \forall M \in \mathbb{N} \forall i v(G(r_i)) > M

    if and only if ..math:

        \forall M \in \mathbb{N} \forall i v(G(r_i)) > M.

    That is, the truncations `r_i` are sufficiently long so that polynomial
    evaluation of `r_i` and `p_i` has the same valuation.

    INPUT:

        - ``v``: list of the valu

        - ``x``: the independent variable in each of the `p`'s

    OUTPUT:
    
        - ``(list)``: a list `[N_1,\ldots,N_n]` such that each `N_i` satisfies
          the condition above.
    """
    n = len(p)
    N = []

    max_Int = max([Int(k,p) for k in xrange(n)])
    for i in xrange(n):
        pairwise_diffs = [valuation(p[k]-p[i],x) for i in xrange(n) if k!=i]
        N.append(max(pairwise_diffs) + max_Int - Int(i,p) + 2)

    return N


def compute_series_truncations(f,x,y,a):
    """
    Computes the Puiseux series expansions at the `x`-point `x=a` with the 
    necessary number of terms in order to compute the integral basis of the 
    algebraic functions field corresponding to `f`.
    """
    # compute the first terms of the Puiseux series expansions
    p = puiseux(f,x,y,a,1,parametric=False)
    
    # compute the expansion bounds
    N = compute_expansion_bounds(p,x)
    Nmax = max(N)

    # compute Puiseux series and truncate using the expansion bounds.
    # [[[XXX]]] this needs to be drastically improved, probably by adding the
    # option to compute Puiseux series up to a certain degree bound
    # instead of by number of terms.
    z = sympy.Symbol('z')
    r = puiseux(f,x,y,a,Nmax,parametric=False)
    n = len(r)
    for i in xrange(n):
        ri = r[i]
        ri = ri.subs(x,z+a)
        ri = ri.expand(mul=True,force=True).series(z) + sympy.O(z**N[i])
        r[i] = ri.removeO().subs(z,x-a)

    return r


def integral_basis(f,x,y):
    """
    Compute the integral basis of the polynomial `f` at the point `x=a`.

    """
    # If the curve is not monic then map y |-> y/lc(x) where lc(x)
    # is the leading coefficient of f
    d  = sympy.degree(f,y)
    lc = sympy.LC(f,y)
    if x in lc:
        f = sympy.ratsimp( f.subs(y,y/lc)*lc**(d-1) )
    else:
        f = f/lc;
    
    # Compute the set of irreducible polynomials k(x) for which 
    # k^2 | Res(f, df/dy)
    p = sympy.Poly(f,[x,y])
    n = p.degree(y)
    res = sympy.resultant(p,p.diff(y),y)
    factors = sympy.factor_list(res)[1]
    df = [k for k,deg in factors if (deg > 1) and (sympy.LC(k) == 1)]

    # Here, r_{k,i} = r[k][i], the ith Puiseux series for the kth
    # factor dividing the above resultant
    alpha = []
    r = []
    for l in range(len(df)):
        k = df[l]
        alphak = sympy.roots(k).keys()[0]  # pick a root of k
        rk = compute_series_truncations(f,x,y,alphak)

        alpha.append(alphak)
        r.append(rk)

    # Main Loop
    b = [1]
    for d in range(1,n):
        # intiial guess for b_d
        bd = y*b[-1]
        a = sympy.symbols('a:%d'%d)
        for l in range(len(df)):
            # get k,alphak data
            k = df[l]
            alphak = alpha[l]
            rk = r[l]

            # create list of indeterminants and loop to compute bd
            found_something = True
            while found_something:
                A = (sum(ak*bk for ak,bk in zip(a,b)) + bd) / (x - alphak)
                #A = A.subs(x,z+alphak)
                # construct system of equations consisting of the coefficients
                # of negative powers of (x-alphak) in the substitutions
                # A(r_{k,1}),...,A(r_{k,n})
                equations = []
                for rkl in rk:
                    # [[XXX]] THE FOLLOWING TERM LOOP IS REALY SLOW
                    lser = A.subs(y,rkl).expand(mult=True,force=True).lseries(x,alphak)
                    for term in lser:
                        coeff,deg = term.as_coeff_exponent(x-alphak)
                        if deg < 0:
                            equations.append(coeff)
                        else:
                            break

                # solve the equations for a0,...,a_{d-1}
                sols = sympy.solve(equations,a)
                if sols is None or sols == []:
                    found_something = False
                else:
                    # this is to check for any free variables
                    if len(sols.keys()) < d:
                        found_something = False
                    else:
                        bdm1 = sum( sols[a[i]]*bk for i,bk in zip(range(d),b) )
                        bd = (bdm1 + bd) / k

        # after traversing V, append the resulting bd to the list of bs
        # b.append( sy.simplify(bd*lc) )
        b.append( sympy.simplify(bd.subs(y,y*lc)) )
    return b




if __name__=="__main__":
    from sympy.abc import x,y,T

    f1 = (x**2 - x + 1)*y**2 - 2*x**2*y + x**4                   # yes *
    f2 = -x**7 + 2*x**3*y + y**3                                 # yes
    f3 = (y**2-x**2)*(x-1)*(2*x-3) - 4*(x**2+y**2-2*x)**2        # yes
    f4 = y**2 + x**3 - x**2                                      # yes
    f5 = (x**2 + y**2)**3 + 3*x**2*y - y**3                      # no (oerr)
    f6 = y**4 - y**2*x + x**2                                    # no (wrong)
    f7 = y**3 - (x**3 + y)**2 + 1

    f8 = (x**6)*y**3 + 2*x**3*y - 1                              # no (err)
    f9 = 2*x**7*y + 2*x**7 + y**3 + 3*y**2 + 3*y                 # yes
    f10= (x**3)*y**4 + 4*x**2*y**2 + 2*x**3*y - 1                # no (wrong)

    f = f10
        
    print "Plane curve..."
    sympy.pprint(f)

    print "\nComputing singular points..."
    p = sympy.Poly(f,[x,y])
    n = p.degree(y)
    res = sympy.resultant(p,p.diff(y),y)
    factors = sympy.factor_list(res)[1]
    df = [k for k,deg in factors if (deg > 1) and (sympy.LC(k) == 1)]
    print "\t resultant =", sympy.factor(res)
    print "\t        df =", df
    print "\t (monic irred. polynomials k s.t. k^2 | res)"

    print "\nTruncated puiseux series expansions"
    alpha = [0]
    for alphak in alpha:
        p = compute_series_truncations(f,x,y,alphak)
        for rk in p:
            sympy.pprint(rk)

    print "\nComputing integral basis..."
    b = integral_basis(f,x,y)
    for bk in b:
        sympy.pretty_print(bk)
        print
    
