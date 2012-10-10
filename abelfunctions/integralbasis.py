"""
Integral Bases
==============

A module for computing integral bases of algebraic functions fields of
the form C[x,y] / ( f(x,y) ). The algorithm is based off of the paper
"An Algorithm for Computing an Integral Basis in an Algebraic Function
Field" by Mark van Hoeij.

Authors
-------

- Chris Swierczewski (Initial Version, October 2012)

References
----------

[vH] Mark van Hoeij. "An Algorithm for Computing an Integral Basis in
an Algebraic Function Field". J. Symbolic Computation. (1994) 18,
p. 353-363

"""
import sympy

from puiseux import puiseux


def valuation(p,x):
    """
    Given a collection of Puiseux series, return the valuations. That
    is, the exponents of the leading order term.
    """
    return p.expand(mult=True,force=True).leadterm(x)[1]

def Int(i,p,x):
    """
    The function .. math:

        Int_i = \sum_{k \neq i} v(p_i - p_k)

    for computing the Puiseux series bounds.
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
    polynomials `G \in L[x,y]` the truncation `r_i` of the Puiseux
    series `p_i` satisfying `v(r_i - p_i) > N_i` satisfies the
    relation ..math:

        \forall M \in \mathbb{N} \forall i v(G(r_i)) > M

    if and only if ..math:

        \forall M \in \mathbb{N} \forall i v(G(r_i)) > M.

    That is, the truncations `r_i` are sufficiently long so that
    polynomial evaluation of `r_i` and `p_i` has the same valuation.
    """
    n = len(p)
    N = []

    max_Int = max([Int(k,p,x) for k in xrange(n)])
    for i in xrange(n):
        pairwise_diffs = [valuation(p[k]-p[i],x) for i in xrange(n) if k!=i]
        N.append(max(pairwise_diffs) + max_Int - Int(i,p,x) + 2)

    return N


def compute_series_truncations(f,x,y,a,T):
    """
    Computes the Puiseux series expansions at the `x`-point `x=a` with
    the necessary number of terms in order to compute the integral
    basis of the algebraic functions field corresponding to `f`. The
    Puiseux series is returned in parametric form for computational
    efficiency. (Sympy doesn't do as well with fractional exponents.)
    """
    # compute the first terms of the Puiseux series expansions
    p = puiseux(f,x,y,a,1,parametric=False)
    
    # compute the expansion bounds
    N = compute_expansion_bounds(p,x)
    Nmax = max(N)

    # compute Puiseux series and truncate using the expansion bounds.
    r = puiseux(f,x,y,a,degree_bound=Nmax,parametric=T)
    n = len(r)

    for i in xrange(n):
        ri_X, ri_Y = r[i]
        ramification_index = sympy.degree(ri_X, T)
        ri_Y = ri_Y + sympy.O( T**(N[i]*ramification_index) )
        r[i] = (ri_X, ri_Y.removeO())

    return list(set(r))


def integral_basis(f,x,y):
    """
    Compute the integral basis {b1, ..., bg} of the algebraic function
    field C[x,y] / (f).
    """
    T = sympy.Symbol('T')

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

    # r[k][i] is the ith Puiseux series (in parametric form) for the
    # kth factor dividing the above resultant
    alpha = []
    r = []
    for l in range(len(df)):
        k = df[l]
        alphak = sympy.roots(k).keys()[0]         # pick a root of k
        rk = compute_series_truncations(f,x,y,alphak, T)

        alpha.append(alphak)
        r.append(rk)

    # Main Loop
    a = sympy.symbols('a:%d'%n)
    b = [1]
    for d in range(1,n):
        # intiial guess for b_d. Uses the trick of 
        bd = y*b[-1]
        for l in range(len(df)):
            # get k,alphak data
            k = df[l]
            alphak = alpha[l]
            rk = r[l]

            # create list of indeterminants and loop to compute bd
            found_something = True
            while found_something:
                A = (sum(ak*bk for ak,bk in zip(a,b)) + bd) / (x - alphak)
                # construct system of equations consisting of the coefficients
                # of negative powers of (x-alphak) in the substitutions
                # A(r_{k,1}),...,A(r_{k,n})
                equations = []

                for rk_X, rk_Y in rk:
                    # solve for T in terms of x to obtain the
                    # coefficients for use in constucting the a_i
                    # equations below
                    #          alpha + mu T**q = x
                    sols = sympy.solve((x-alphak)-rk_X,T)
                    consts = map(lambda s: s.as_coeff_exponent(x-alphak)[0],
                                 sols)
                    
                    # compute the series in T up to constant order (we
                    # only need the coefficients of the T terms with
                    # negative exponent
                    ser = A.subs([(x,rk_X),(y,rk_Y)]).expand() + sympy.O(1)
                    ser = ser.removeO()
                        
                    # use the constants computed above to generate the
                    # equations that need to be solved.
                    equations.extend([ser.subs(T,c) for c in consts])

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
                        
        # bd found. Append to list of basis elements
        b.append( bd )

    # finally, convert back to singularized curve if necessary
    for i in xrange(1,len(b)):
        b[i] = b[i].subs(y,y*lc)

    return b




if __name__=="__main__":
    from sympy.abc import x,y,T
#    import cProfile, pstats

    f1 = (x**2 - x + 1)*y**2 - 2*x**2*y + x**4
    f2 = -x**7 + 2*x**3*y + y**3
    f3 = (y**2-x**2)*(x-1)*(2*x-3) - 4*(x**2+y**2-2*x)**2
    f4 = y**2 + x**3 - x**2
    f5 = (x**2 + y**2)**3 + 3*x**2*y - y**3
    f6 = y**4 - y**2*x + x**2
    f7 = y**3 - (x**3 + y)**2 + 1
    f8 = (x**6)*y**3 + 2*x**3*y - 1
    f9 = 2*x**7*y + 2*x**7 + y**3 + 3*y**2 + 3*y
    f10= (x**3)*y**4 + 4*x**2*y**2 + 2*x**3*y - 1

    f = f5
        
    print "Plane curve...\n"
    sympy.pprint(f)

    print "\nComputing Puiseux series (for reference)\n"
    r = compute_series_truncations(f,x,y,0,T)
    for ri in r: sympy.pprint(ri)

    print "\nComputing integral basis...\n"
    b = integral_basis(f,x,y)
    sympy.pprint(b)

#    cProfile.run("b = integral_basis(f,x,y)",'intbasis.profile')
#     p = pstats.Stats('intbasis.profile')
#     p.strip_dirs()
#     p.sort_stats('time').print_stats(12)
#     p.sort_stats('cumulative').print_stats(12)
#     p.sort_stats('calls').print_stats(12)
    

    
