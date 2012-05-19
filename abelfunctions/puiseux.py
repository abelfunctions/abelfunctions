"""
Puiseux Series
==============

Tools for computing Puiseux series. A necessary component to computing 
integral bases and with Riemann surfaces.
"""

import sympy
from operator import itemgetter

import pdb


def _coefficient(F):
    """
    Helper function. Returns a dictionary of coefficients of the polynomial
    indexed by monomial powers ..math :

        `a_{ij} X^j Y^i`

    INPUTS:
    
    -- ``F``: a sympy polynomial

    OUTPUTS:

    -- ``dict``: a dictionary such that ``d[(i,j)] = a_ij``.

    EXAMPLES:

        >>> from sympy import Poly
        >>> from sympy.abc import x,y
        >>> f = Poly(y**2 + x**2*(x+1))
        >>> _coefficient(f) == {(0, 2): -1, (0, 3): -1, (2, 0): 1}
        True
    """
    # compute useful dictionary of coefficients indexed by the support of 
    # the polynomial
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
    q,mu,m,beta,eta = tau
    Xnew = sympy.Poly(mu*X**q,X)
    Ynew = X**m*(beta+eta*Y)
    Fnew = sympy.Poly( sympy.compose(sympy.compose(F,Xnew),Ynew,Y), X,Y)
    Fnew = sympy.polytools.pquo(Fnew, sympy.Poly(X**l,X), domain=Fnew.domain)
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
    if type(hull) == sympy.Segment:
        hull_with_bdry = support       # colinear support is a newton polygon
    else:
        hull_with_bdry = [p for p in support if not hull.encloses(sympy.Point(p))]
    newton = []

    # find the start and end points (0,J) and (I,0). Include points along
    # the i-axis if I==1. Otherwise, only include points with negative
    # slope if I==2
    JJ = min([j for (i,j) in hull if i == 0])
    if I == 2: II = min([i for (i,j) in hull if j == 0])
    else:      II = max([i for (i,j) in hull if j == 0])
    testslope = -float(JJ)/II

    # determine largest slope with (0,JJ). If this is greater than the test
    # slope then there exist points above the line connecting (0,JJ) with 
    # (II,0) this fact is used to deal with certain borderline cases
    include_borderline_colinear = (type(hull) == sympy.Segment)
    for (i,j) in hull_with_bdry:
        # when the point is on the j-axis, the onle one we want is the
        # point (0,JJ)
        if i == 0:
            if j == JJ: slope = -sympy.oo
            else:       slope = sympy.oo
        else:           slope = float(j-JJ)/i     

        if slope > testslope: 
            include_borderline_colinear = True
            break

    # loop through all points on the boundary and determine if it's in the
    # newton polygon using a testslope method
    for (i,j) in hull_with_bdry:
        # when the point is on the j-axis, the onle one we want is the
        # point (0,JJ)
        if i == 0:
            if j == JJ: slope = -sympy.oo
            else:       slope = sympy.oo
        else:           slope = float(j-JJ)/i     
        
        # if the slope is less than the test slope or if we're in the case
        # where we include points along the j=0 line then add the point
        # to the newton polygon
        if (slope < testslope) or (I==1 and j==0): 
            newton.append((i,j))
        elif (slope == testslope) and include_borderline_colinear:
            # borderline case is when there is only one segment
            # from (0,JJ) to (II,0). When this is the case, include all
            # points whose slope matches testslope
            newton.append((i,j))

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

        # check against all following points for colinearity by comparing
        # slopes. append all colinear points to side
        k = 1
        for pt in newton[n+2:]:
            slope = float(pt[1]-side[0][1]) / (pt[0]-side[0][0])
            if abs(slope - sideslope) < eps:
                side.append(pt)
                k += 1
            else:
                # when we reach the end of the newton polygon we need
                # to shift the value of k a little bit so that the 
                # last side is correctly captured
#                if k == N-n-1: k -= 1
                break
        n += k

        # compute q,m,l such that qj + mi = l and Phi
        q = sideslope.q
        m = -sideslope.p
        l = q*side[0][1] + m*side[0][0]
        i0 = min(side, key=itemgetter(0))[0]
        Phi = sum(a[(i,j)]*Z**sympy.Rational(i-i0,q) for (i,j) in side)
        Phi = sympy.Poly(Phi,Z)

        params.append((q,m,l,Phi))

    return params    


def newton(F,X,Y,H,version):
    """
    Compute the Puiseux series data `\pi = (\tau_1,\ldots,\tau_R)` where 
    `\tau_h = (q_h,\mu_h,m_h,\beta_h)`.
    """
    K = sympy.QQ
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
        # grow each expansion to the number of desired terms
        # TODO: at some point, change this to computing the degree
        while len(pi) < H:
            # if the set of all (0,j), j!=0 is empty, then we've 
            # encountered a finite puiseux expansion
            a = dict(F.terms())
            ms = [j for (j,i) in a.keys() if i==0 and j!=0]
            if ms == []: break
            else:        m = min(ms)
            beta = - a[(m,0)]/a[(0,1)]
            tau = (1,1,m,beta,1)
            pi.append(tau)
            F = _new_polynomial(F,X,Y,tau,m)

        R.append(pi)

    return R
        


def singular(F,X,Y,L,pi,version):
    """
    Computes a collection of pairs `(\pi_1,F_1)` where `\pi_1` is a finite
    `\mathbb{K}`-expansion beginning by `\pi` and `F_1 \in\mathbb{L}_1[X,Y]`
    for some extension `\mathbb{L}_1` of `\mathbb{L}`.

    """
    S = []
    if pi == []: I = 1
    else:        I = 2

    for (tau,l,r) in singular_term(F,X,Y,L,I,version):
        pi1 = pi + [tau]
        F1 = _new_polynomial(F,X,Y,tau,l)
        L1 = F1.domain
        if r == 1: S.append((pi1,F1))
        else:      S.extend(singular(F1,X,Y,L1,pi1,version))

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
        Z = Phi.gen
        for (Psi,r) in _square_free(Phi):
            Psi = sympy.Poly(Psi,Z)
            
            for xi,M in sympy.roots(Psi).iteritems():
                # the classical version returns the "raw" roots
                if version == 'classical':
                    P = sympy.Poly(U**q-xi,U)
                    beta = RootOf(P,0,radicals=False)
                    tau = (q,1,m,beta,1)
                    T.append((tau,l,r))
                # the rational version rescales parameters so as to 
                # include ony rational terms in the Puiseux expansions.
                if version == 'rational':
                    mu = xi**(-v)
                    beta = xi**u
                    tau = (q,mu,m,beta,1)
                    T.append((tau,l,r))

    return T


def is_singular(f,x,y):
    """
    Determines if `f` is singular at the `x`-point `x=0`.

    Inputs:

    - ``f``: a plane algebraic curve
    - ``x,y``: variables

    Outputs:

    - (bool): ``True`` if `f` is singular

    Examples:

        >>> 1+1
        2
    """
    p = sympy.Poly(f,x,y)
    coeffs = _coefficient(p)
    deg = p.degree(y)

    # the expansion is singular if there is no "c y**deg" where c is constant
    # and if there is a constant term
    sing_coeffs = [(i,j) for (i,j),a in coeffs.iteritems() if i==deg]
    if (0,0) in coeffs.keys() and (deg,0) not in sing_coeffs:
        return True
    else:
        return False

    
    
def desingularize(f,x,y,version=version):
    """
    If f is singular, it is desginularized. Outputs new f and Puiseux 
    series expansion data.

    Inputs:

    - ``f``: a plane algebraic curve
    - ``x,y``: variables

    Outputs:

    - ``f``: the desingularized curve
    - ``pi_singular``: Puiseux series data `\pi=(\tau_1,\ldots,\tau_n)`

    Examples:
    
        >>> 1+1
        2
    """
    pis = []
    if is_singular(f,x,y):
        coeffs = _coefficient(f)
        c = coeffs.pop((0,0))

        # for each monomial c x**j y**i find the dominant term: that is
        # the one that cancels out the constant term c. This is done
        # by substituting x = T**q, y = eta T**m giving the monomial 
        # c eta**iT**(qj+mi). To balance the equation (kill the constant term) 
        # we need qj+mi=0. q = i, m = -j satisfies this equation.
        #
        # Finally, we need to check that this choice of q,m doesn't introduce 
        # terms with negative exponent in the curve.
        q,m = (1,1)
        for (i,j),aij in coeffs.iteritems():
            # compute q,m
            g = sympy.gcd(i,j)
            q = sympy.Rational(i,g)
            m = -sympy.Rational(j,g)

            # check if the other terms remain positive.
            if all(q*jj+m*ii>=0 for ii,jj in coeffs.keys()):
                break

        if (q,m) == (1,1):
            raise ValueError("Unable to compute singular term.")            
        
        # now compute the values of eta that cancel the constant term c
        p = [aij*x**i for (i,j) in coeffs.keys() if q*j+m*i == 0]
        p = sympy.Poly(sum(p) + c, x)

        pdb.set_trace()
        
        # compute the transformation data that we will prepend to the
        # series construction process below and transform the polynomial.
        for eta,M in sympy.roots(p).iteritems():
            tau_singular = (q,1,m,1,eta)
            pi = [tau_singular]
            f = _new_polynomial(f,x,y,tau_singular,0)
            pi.
#            ff, ppis = desingularize(f,x,y)
#            pis.extend(ppis)

    return f, pis


def build_series(pis,x,y,T,a,parametric):
    """
    Builds all Puiseux series from pi data.

    `\pi_i=(q,\mu,m,\beta,\eta), X \mapsto\mu X^q,Y \mapsto X^m(\beta+\eta Y)`
    """
    series = []
    # build singular part of expansion series
    for pi in pis:
        # get first elements
        q,mu,m,beta,eta = pi[0]
        P = mu*x**q
        Q = (beta + y)*x**m
        
        # build rest of series
        for h in xrange(1,len(pi)):
            q,mu,m,beta,eta = pi[h]
            P1 = mu*x**q
            Q1 = (beta + y)*x**m
            
            P = P.subs(x,P1)
            Q = Q.subs([(x,P1),(y,Q1)])

        P = P.subs([(x,T),(y,0)]).expand()
        Q = Q.subs([(x,T),(y,0)]).expand()

        # append parametric or non-parametric form
        if parametric:
            if a == sympy.oo: series.append((1/P,Q))
            else:             series.append((P+a,Q))
        else:
            if a == sympy.oo: solns = sympy.solve(1/x-P,T)
            else:  
                solns = sympy.solve((x-a)-P,T)
                for TT in solns:  series.append(Q.subs(T,TT))

    return series




def puiseux(f, x, y, a, n, parametric=True, version='rational'):
    """
    Computes the first `n` terms of the Puiseux series expansions of 
    `f = f(x,y)` at `x=a`.

    INPUT:

    -- ``parametric``: (default: ``True``) If set to ``True``, returns 
       parameterized form of the Puiseux series expansions. That is, each 
       series is a pair `x = x(T), y = y(T)`. If set to ``False``, returns 
       unparameterized Puiseux series expansions `y = y(x)`.

    -- ``version``: (default: ``'rational'``) use 'rational' for rational
       Puiseux series expansions. Use 'classical' for expansions containing
       algebraic numbers
    """
    if version not in ['classical','rational']:
        raise AttributeError("'version' must be 'classical' or 'rational'")

    f = sympy.Poly(f,x,y)

    # scale f accordingly
    if a == sympy.oo: 
        f = (f.subs(x,1/x) * x**f.deg(x)).expand()
    else:
        f = f.subs(x,x+a)
        
    # desingularize if necessary
    if is_singular(f,x,y):
        pis = desingularize(f,x,y,version=version)
    else:
        pis = newton(f,x,y,n,version=version)

    # compute the puiseux series
    T = sympy.Symbol('T')
    series = build_series(pis,x,y,T,a,parametric)
    
    return series
            
"""
TESTS
"""
if __name__ == "__main__":
    print "==== Module Test: puiseux.py ==="
    from sympy.abc import x,y,T
    import cProfile, pstats

    f1 = (x**2 - x + 1)*y**2 - 2*x**2*y + x**4                   # check
    f2 = -x**7 + 2*x**3*y + y**3                                 # check
    f3 = (y**2-x**2)*(x-1)*(2*x-3) - 4*(x**2+y**2-2*x)**2        # check
    f4 = y**2 + x**3 - x**2                                      # check
    f5 = (x**2 + y**2)**3 + 3*x**2*y - y**3                      # check
    f6 = y**4 - y**2*x + x**2                                    # check
    f7 = y**3 - (x**3 + y)**2 + 1

    f8 = (x**6)*y**3 + 2*x**3*y - 1
    f9 = 2*x**7*y + 2*x**7 + y**3 + 3*y**2 + 3*y
    f10= (x**3)*y**4 + 4*x**2*y**2 + 2*x**3*y - 1

    f  = f8
    a  = 0
    N  = 2

    print "Curve:\n"
    sympy.pretty_print(f)
    
#     cProfile.run("P = puiseux(f,x,y,a,N,parametric=True,version='rational')",'puiseux.profile')
#     p = pstats.Stats('puiseux.profile')
#     p.sort_stats('time').print_stats(10)
#     p.sort_stats('cumulative').print_stats(10)
#     p.sort_stats('calls').print_stats(10)

    P = puiseux(f,x,y,a,N,parametric=True,version='rational')

    print "\nPuiseux Expansions at x =", a
    for Y in P:
        print "Expansion:"
#        print "X ="
#        sympy.pretty_print(X)
        print "\nY ="
        sympy.pretty_print(Y)
        print

