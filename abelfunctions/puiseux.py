"""
Puiseux Series
==============

Tools for computing Puiseux series. A necessary component to computing 
integral bases and with Riemann surfaces.
"""

import sympy
import pdb
from operator import itemgetter


def _coefficient(F):
    """
    Helper function. Returns a dictionary of coefficients of the polynomial
    indexed by monomial powers ..math :

        `a_{ij} X^j Y^i`

    INPUTS:
    
        -- ``F``: a sympy polynomial

    OUTPUTS:

        -- ``dict``: a dictionary such that ``d[(i,j)] = a_ij``.
    """
    # compute useful dictionary of coefficients indexed
    # by the support of the polynomial
    d = {}
    monoms = F.monoms()
    coeffs = F.coeffs()
    for a,(j,i) in zip(coeffs,monoms):   # lexicographic ordering
        d[(i,j)] = a
    return d



def _bezout(q,m):
    """
    Given two coprime integers `q,m` with `q>0` returns `u,v` such that
    `uq+mv=1`.
    """
    u,v,g = sympy.gcdex(q,m)
    if u*q+v*m != 1: raise ValueError("Bezout algorithm failed.")
    return u,v

def _square_free(Phi):
    """
    Given a polynomial `\Phi \in \mathbb{L}[Z]` returns a collection of pairs
    `\{(\Psi,r)\}` with `\Psi \in \mathbb{L}[Z]` and `r` a positive integer
    such that each `\Psi` is square-free and `\Phi = \prod \Psi^r` and the
    `\Psi`'s are pairwise coprime.

    ALGORITHM:

    Such a decomposition can be obtained with derivations and gcd computations
    without any factorization algorithm. It's implemented in sympy as 
    ``sympy.sqf`` and ``sympy.sqf_list``
    """
    return sympy.sqf_list(Phi)[1]


def _new_polynomial(F,X,Y,tau,l):
    """
    Computes the next iterate of the newton-puiseux algorithms. Given the 
    Puiseux data `\tau = (q,\mu,m,\beta)` ``_new_polynomial`` returns .. math:
    
        \tilde{F} = F(\mu X^q,X^m(\beta+Y))/X \in \mathbb{L}(\mu,\beta)(X)[Y]

    In this algorithm, the choice of parameters will always result in .. math:

        \tilde{F} \in \mathbb{L}(\mu,\beta)[X,Y]
    """
    q,mu,m,beta = tau
    Fnew = F.subs({X:(mu*X**q), Y:(X**m*(beta+Y))})
    Fnew = (Fnew/(X**l)).expand()
    return Fnew


def polygon(F,X,Y,I):
    """
    Computes a set of parameters and polynomials in one-to-one correspondence
    with the segments fo the Newton polygon of F. If ``I=2`` the 
    correspondence is only with the segments with negative slope.

    The segment `\Delta` corresponding to the list `(q,m,l,\Phi)` is on the
    line `qj+mi=l` in the `(i,j)`-plane and .. math:

        \Phi = \sum_{(i,j) \in \Delta} a_{ij}Z^{(i-i_0)/q}

    where `i_0` is the smallest value of `i` such that there is a point
    `(i,j) \in \Delta`. Note that `\Phi \in \mathbb{L}[Z]`.


    INPUTS:
    
        -- ``F``: a polynomial in `\mathbb{L}[X,Y]`
        
        -- ``X,Y``: the variable defining ``F``
        
        -- ``I``: a parameter 

    OUTPUTS:
    
        -- ``(list)``: a list of tuples `(q,m,l,\Phi)` where `q,m,l` are integers with `(q,m) = 1`, `q>0`, and `\Phi \in \mathbb{L}[Z]`.
    """
    # compute the coefficients and support of F
    P = sympy.poly(F,X,Y)
    a = _coefficient(P)
    support = a.keys()

    # compute the lower convex hull of F.
    #
    # since sympy.convex_hull doesn't include points on edges, we need
    # to compare back to the support. the convex hull will contain all
    # points. Those on the boundary are considered "outside" by sympy.
    hull = sympy.convex_hull(*support)
    hull_with_bdry = [p for p in support if not hull.encloses(sympy.Point(p))]
    newton = []

    # find the start and end points (0,J) and (I,0)
    JJ = min([j for (i,j) in hull if i == 0])
    if I==2: 
        II = min([i for (i,j) in hull if j == 0])
    else:    
        II = max([i for (i,j) in hull if j == 0])
    testslope = -float(JJ)/II

    for (i,j) in hull_with_bdry:
        # when the point is on the j-axis, the onle one we want is the
        # point (0,JJ)
        if i == 0:
            if j == JJ: slope = -sympy.oo
            else:       slope = sympy.oo
        else:           slope = float(j-JJ)/i     

        if slope < testslope: 
            newton.append((i,j))
        elif (slope == testslope):
            # in the case of slope equality we only include the points on the
            # i-axis. This eliminates the case of colinear points on the 
            # outer face of the support from (0,JJ) to (II,0)
            if j == 0: newton.append((i,j))

    newton.sort(key=itemgetter(1),reverse=True) # sort second in j-th coord
    newton.sort(key=itemgetter(0))              # sort first in i-th coord

    #
    # now that we have the newton polygon we compute the parameters
    # (q,m,l,Phi) for each side Delta on the polygon.
    #
    Z = sympy.Symbol('Z')
    params = []
    eps = 1e-14
    N = len(newton)
    n = 0
    while n < N-1:
        # determine slope of current line
        side = [newton[n],newton[n+1]]
        sideslope = sympy.Rational(side[1][1]-side[0][1],side[1][0]-side[0][0])
        k = 2

        # check against all following points for colinearity by comparing
        # slopes. append all colinear points to side
        for kk in xrange(2,N-n-1):
            pt = newton[n+k]
            slope = float(pt[1]-side[0][1]) / (pt[0]-side[0][0])
            if abs(slope - sideslope) < eps:
                side.append(pt)
            else:
                k = kk
                break
        n += k-1

        # compute q,m,l such that qj + mi = l and Phi
        q = sideslope.q
        m = -sideslope.p
        l = q*side[0][1] + m*side[0][0]
        i0 = min(side, key=itemgetter(0))[0]
        Phi = sum(a[(i,j)]*Z**sympy.Rational(i-i0,q) for (i,j) in side)
        params.append((q,m,l,Phi))

    return params

    

def newton(F,X,Y,H,version='rational'):
    """

    """
    K = None
    if version not in ['classical','rational']:
        raise AttributeError("'version' must be 'classical' or 'rational'")
    return regular(singular(F,X,Y,K,[],version),X,Y,H)


def regular(S,X,Y,H):
    """
    INPUT:

      -- ``S``: a finite set of pairs `\{(\pi_k,F_k)\}_{1 \leq k \leq B}`
         where `\pi_k` is a finite `\mathbb{K}`-expansion, 
         `F_k \in \bar{\mathbb{K}}[X,Y]` with `F_k(0,0) = 0`, 
         `\partial_Y F_k(0,0) \neq 0`, and `F_k(X,0) \neq 0`.

      -- ``H``: a positive integer 

    OUTPUT:
    
      -- ``(list)``: a set `\{ \pi_k' \}_{1 \leq k \leq B}` of finite 
         `\mathbb{K}` expansions such that `\pi_k'` begins with `\pi_k`
         and contains at least `H` `\mathbb{K}`-terms.

    """
    R = []
    for (pi,F) in S:
        P = sympy.poly(F.expand(),X,Y)
        a = _coefficient(P)
        
        # grow each expansion to the number of desired terms
        # TODO: at some point, change this to computing the degree
        while len(pi) < H:
            m = min(j for (i,j) in a.keys() if i==0 and j!=0)
            beta = -sympy.Rational(a[(0,m)], a[(1,0)])
            tau = (1,1,m,beta)
            pi.append(tau)
            F = _new_polynomial(F,X,Y,tau,m)

        R.append(pi)

    return R
        


def singular(F,X,Y,L,pi,version):
    """
    Computes a collection of pairs `(\pi_1,F_1)` where `\pi_1` is a finite
    `\mathbb{K}`-expansion beginning by `\pi` and `F_1 \in\mathbb{L}_1[X,Y]`
    for some extension `\mathbb{L}_1` of `\mathbb{L}`.

    INPUT:

      -- ``F``:

      -- ``L``:
      
      -- ``pi``:

      -- ``version``:

    OUTPUT:

      -- ``(list)``: 

    """
    S = []
    if pi == []: I = 1
    else:        I = 2

    for (tau,l,r) in singular_term(F,X,Y,L,I,version):
        pi1 = pi + [tau]
        F1 = _new_polynomial(F,X,Y,tau,l)
        #L1 = the extension of L generated by the element mu and beta of tau
        L1 = None
        if r == 1: S.append((pi1,F1))
        else:      S.append(singular(F1,X,Y,L1,pi,version))

    return S

def singular_term(F,X,Y,L,I,version):
    """
    Computes a single set of singular terms of the Puiseux expansions.

    For `I=1`, the function computes the first term of each finite 
    `\mathbb{K}` expansion. For `I=2`, it computes the other terms of the
    expansion.
    """
    T = []
    U = sympy.Symbol('U')

    # each side of the newton polygon corresponds to a K-term.
    for (q,m,l,Phi) in polygon(F,X,Y,I):
        # the rational method
        if version == 'rational':
            u,v = _bezout(q,m)

        # each newton polygon side has a characteristic polynomial. For each
        # square-free factor, each root corresponds to a K-term
        for (Psi,r) in _square_free(Phi):
            for (xi, M) in sympy.roots(Psi).iteritems():
                # the classical version returns the "raw" roots
                if version == 'classical':
                    for beta in sympy.roots(U**q-xi).keys():
                        tau = (q,1,m,beta)
                        T.append((tau,l,r))
                # the rational version rescales parameters so as to 
                # include ony rational terms in the Puiseux expansions.
                if version == 'rational':
                    mu = xi**(-v); mu = sympy.Rational(str(mu))
                    beta = xi**u;  beta = sympy.Rational(str(beta))
                    tau = (q,mu,m,beta)
                    T.append((tau,l,r))
    return T
    


"""
TESTS
"""
if __name__ == "__main__":
    print "==== Test Suite: puiseux.py ==="
    # example algebraic curve
    x,y,T = sympy.symbols('x,y,T')
    f1 = -y**5 + (2*x-1)*y**4 - (3*x-x**2)*y**3 - (x-3*x**2)*y**2 + \
         (x**3-2*x**2)*y + x**6
    f2 = y**2 - x**2*(x+1)
    f3 = y-x
    f  = f2

    print "Curve:"
    print 
    print "\t", f
    print

    pis = newton(f,x,y,4,version='rational')
    print "\nPuiseux Expansions:"
    for pi in pis:
        print "Expansion:"
        for h in xrange(len(pi)):
            tau = pi[h]
            print "\ttau =", tau
            print "\t\tX_%d = %s*X_%d**%s" %(h,tau[1],h+1,tau[0])
            print "\t\tY_%d = (%s + Y_%d)X_%d**%s" %(h,tau[3],h+1,h+1,tau[2])
        print
    

