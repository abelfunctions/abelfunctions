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
from utilities import cached_function

import pdb


def valuation(p,x,alpha):
    """
    Given a collection of Puiseux series, return the valuations. That
    is, the exponents of the leading order term.
    """
    terms = p.collect(x-alpha,evaluate=False).keys()
    lead = terms[0]
    val = lead.as_coeff_exponent(x)[1]
    return val


def Int(i,p,x,alpha):
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
            val += valuation(pi-p[k],x,alpha)

    return val


def compute_expansion_bounds(p,x,alpha):
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

    max_Int = max([Int(k,p,x,alpha) for k in xrange(n)])
    for i in xrange(n):
        pairwise_diffs = [valuation(p[k]-p[i],x,alpha) 
                          for i in xrange(n) if k!=i]
        Ni = max(pairwise_diffs) + max_Int - Int(i,p,x,alpha) + 1
        N.append(Ni)

    return N


def compute_series_truncations(f,x,y,alpha,T):
    """
    Computes the Puiseux series expansions at the `x`-point `x=a` with
    the necessary number of terms in order to compute the integral
    basis of the algebraic functions field corresponding to `f`. The
    Puiseux series is returned in parametric form for computational
    efficiency. (Sympy doesn't do as well with fractional exponents.)
    """
    # compute the first terms of the Puiseux series expansions
    p = puiseux(f,x,y,alpha,nterms=0,parametric=False)

    # compute the expansion bounds
    N = compute_expansion_bounds(p,x,alpha)
    Nmax = max(N)

    # compute Puiseux series and truncate using the expansion bounds.
    r = puiseux(f,x,y,alpha,degree_bound=Nmax)
    n = len(r)

    for i in xrange(n):
        terms = r[i].collect(x-alpha,evaluate=False)
        ri_trunc = sum( coeff*term for term,coeff in terms.iteritems()
                        if term.as_coeff_exponent(x)[1] <= N[i] )
        r[i] = ri_trunc

    return r



def _negative_power_coeffs(expr, var, alpha):
    """
    Return a list of the (symbolic) coefficients of expr as an expression
    in var. That is, if

        expr = c1 (x-alpha)**n_1 + ... + cm (x-alpha)**n_m + O(1)

    then the list [c_1, ..., c_m] is returned. That is, the coefficients of 
    the negative powers of expr.

    NOTE: This is written because sympy's built in functions don't seem to do
    what I want it to do.
    """
    expr = expr.subs(var,var+alpha).expand(log=False,power_base=False,
                                           power_exp=False,multinomial=True,
                                           basic=False,force=True)
    terms  = expr.collect(var,evaluate=False)
    coeffs = [coeff for term,coeff in terms.iteritems() 
              if term.as_coeff_exponent(var)[1] < 0]
    return coeffs


@cached_function
def integral_basis(f,x,y):
    """
    Compute the integral basis {b1, ..., bg} of the algebraic function
    field C[x,y] / (f).
    """
    # If the curve is not monic then map y |-> y/lc(x) where lc(x)
    # is the leading coefficient of f
    T  = sympy.Dummy('T')
    d  = sympy.degree(f,y)
    lc = sympy.LC(f,y)
    if x in lc:
        f = sympy.ratsimp( f.subs(y,y/lc)*lc**(d-1) )
    else:
        f = f/lc
        lc = 1

    #
    # Compute df
    #
    p = sympy.Poly(f,[x,y])
    n = p.degree(y)
    res = sympy.resultant(p,p.diff(y),y)
    factors = sympy.factor_list(res)[1]
    df = [k for k,deg in factors if (deg > 1) and (sympy.LC(k) == 1)]

    #
    # Compute series truncations at appropriate x points
    #
    alpha = []
    r = []
    for l in range(len(df)):
        k = df[l]
        alphak = sympy.roots(k).keys()[0]
        rk = compute_series_truncations(f,x,y,alphak,T)
        alpha.append(alphak)
        r.append(rk)

    #
    # Main Loop
    #
    a = [sympy.Dummy('a%d'%k) for k in xrange(n)]
    b = [1]
    for d in range(1,n):
        bd = y*b[-1]

        for l in range(len(df)):
            k = df[l]
            alphak = alpha[l]
            rk = r[l]

            found_something = True            
            while found_something:
                # construct system of equations consisting of the coefficients
                # of negative powers of (x-alphak) in the substitutions
                # A(r_{k,1}),...,A(r_{k,n})
                A  = (sum(ak*bk for ak,bk in zip(a,b)) + bd) / (x - alphak)

                coeffs = []
                for rki in rk:
                    # substitute and extract coefficients
                    A_rki  = A.subs(y,rki)
                    coeffs.extend(_negative_power_coeffs(A_rki, x, alphak))
               
                # solve the coefficient equations for a0,...,a_{d-1}
                coeffs = [coeff.as_numer_denom()[0] for coeff in coeffs]
                sols = sympy.solve_poly_system(coeffs, a[:d])
                if sols is None or sols == []:
                    found_something = False
                else:
                    sol  = sols[0]
                    bdm1 = sum( sol[i]*bk for i,bk in zip(range(d),b) )
                    bd   = (bdm1 + bd) / k
                        
        # bd found. Append to list of basis elements
        b.append( bd )

    # finally, convert back to singularized curve if necessary
    for i in xrange(1,len(b)):
        b[i] = b[i].subs(y,y*lc).ratsimp()

    return b




if __name__=="__main__":
    from sympy.abc import x,y,T
    import cProfile, pstats

    f1 = (x**2 - x + 1)*y**2 - 2*x**2*y + x**4
    f2 = -x**7 + 2*x**3*y + y**3
#    f3 = (y**2-x**2)*(x-1)*(2*x-3) - 4*(x**2+y**2-2*x)**2
    f3 = y**2 - (x+1)*(x-1)*(x-2)*(x+2)*(x-3)*(x+3)*(x-4)*(x+4)
    f4 = y**2 + x**3 - x**2
    f5 = (x**2 + y**2)**3 + 3*x**2*y - y**3
    f6 = y**4 - y**2*x + x**2
    f7 = y**3 - (x**3 + y)**2 + 1
    f8 = (x**6)*y**3 + 2*x**3*y - 1
    f9 = 2*x**7*y + 2*x**7 + y**3 + 3*y**2 + 3*y
    f10= (x**3)*y**4 + 4*x**2*y**2 + 2*x**3*y - 1

    fs = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10]

    for f in fs:
        print "Plane curve...\n"
        sympy.pprint(f)

        print "\nComputing integral basis...\n"
        b = integral_basis(f,x,y)
        sympy.pprint(b)
    
#     f = f5

#     cProfile.run("b = integral_basis(f,x,y)",'intbasis.profile')
#     p = pstats.Stats('intbasis.profile')
#     p.strip_dirs()
#     p.sort_stats('time').print_stats(12)
#     p.sort_stats('cumulative').print_stats(12)
#     p.sort_stats('calls').print_stats(12)
