import sympy
import pdb 

from abelfunctions import puiseux


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
        N.append(max(pairwise_diffs) + max_Int - Int(i,p) + 1)

    return N


def compute_series_truncations(f,x,y,a):
    """
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
    r = puiseux(f,x,y,a,Nmax,parametric=False)
    n = len(r)
    for i in xrange(n):
        ri = r[i]
        r[i] = ri.expand(mult=True,force=True) + sympy.O(x**N[i])

    return r


def integral_basis(f,x,y,a):
    """
    Compute the integral basis of the polynomial `f` at the point `x=a`.

    """
    # 1)
    p = sympy.Poly(f,[x,y])
    n = p.degree(y)
    res = sympy.resultant(p,p.diff(y),y)
    factors = sympy.sqf_list(res)[1]
    df = [k for k,deg in factors if deg > 1]

    # 2) Here, r_{k,i} = r[k][i]
    alpha = []
    r = []
    for l in range(len(df)):
        k = df[l]
        alphak = sympy.roots(k).keys()[0]  # pick a root of k
        rk = compute_series_truncations(f,x,y,alpha)

        alpha.append(alphak)
        r.append(rk)
    
    # 3)
    b = [1]
    for d in range(1,n):
        # intiial guess for b_d
        bd = y*b[-1]
        for l in range(len(df)):
            # get k,alphak data
            k = df[l]
            alphak = alpha[l]

            # loop to compute bd
            found_something = True
            while found_something:
                # create list of indeterminants
                a = sympy.symbols('a:%d'%d)
                A = (sum(ak*bk for ak,bk in zip(a,b)) + bd)/(x-alphak)
                
                # construct system of equations consisting of the coefficients
                # of negative powers of (x-alphak) in the substitutions
                # A(r_{k,1}),...,A(r_{k,n})
                
                # solve the equations for a0,...,a_{d-1}
                sols = sympy.solve(equations,a)
                
                # XXX be careful here. sympy will return stuff if there
                # are infinitely many solutions
                if sols:
                    sol = sols[0]
                    for i in xrange(d):
                        ai = sol[a[i]].subs(alphak,x)
                        
                        

     
    return r




if __name__=="__main__":
    from sympy.abc import x,y,T

    f = -x**7 + 2*x**3*y + y**3

    print "Computing expansion bounds..."
    p = puiseux(f,x,y,0,1,parametric=False)
    pdb.set_trace()
    N = compute_expansion_bounds(p,x)
    n = len(p)
    for i in xrange(n):
        print "\tSeries #%d:"%i
        sympy.pretty_print(p[i])
        print "\tValuation:        %s"%(valuation(p[i],x))
        print "\tTruncation order: %s\n"%(N[i])
        
    print "\nComputing series truncations..."
    r = compute_series_truncations(f,x,y,0)
    for ri in r:
        sympy.pretty_print(ri)
    
