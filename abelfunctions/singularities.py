"""
Singularities
=============

A module for computing the singular points of a complex plane algebraic curve
including their multiplicities, branching numbers, multiplicities, and delta
invariants.

Authors
-------

- Chris Swierczewski (initial version, October 2012)

"""
import pdb

import sympy

from puiseux import puiseux
from integralbasis import Int

# temporary, hidden symbol to maintain clean Sympy cache
_t = sympy.Symbol('t')
_z = sympy.Symbol('z')

def homogenize(f,x,y,z):
    """
    Returns the polynomial in homogenous coordinates.
    """
    p = sympy.poly(f,[x,y])
    d = max(map(sum,p.monoms()))
    F = sympy.expand( z**d * f.subs([(x,x/z),(y,y/z)]) )
    return F, d


def _singular_points_finite(f,x,y):
    """
    Returns the finite singular points of f.
    """
    S = []

    # compute the finite singularities: use the resultant
    p  = sympy.Poly(f,[x,y])
    n  = p.degree(y)
    res = sympy.Poly(sympy.resultant(p,p.diff(y),y),x)
    for xk,deg in res.all_roots(multiple=False,radicals=False):
        if deg > 1:
            fxk = sympy.Poly(f.subs({x:xk}), y)
            for ykj,_ in fxk.all_roots(multiple=False,radicals=False):
                fx = f.diff(x)
                fy = f.diff(y)
                subs = {x:xk,y:ykj}
                if (fx.subs(subs) == 0) and (fy.subs(subs) == 0):
                    S.append((xk,ykj,1))

    return S


def _singular_points_infinite(f,x,y):
    """
    Returns the singular points of f at infinity.
    """
    S = []

    # compute homogenous polynomial
    F, d = homogenize(f,x,y,_z)

    # find singular points at infinity
    F0 = sympy.Poly(F.subs([(_z,0)]),[x,y])
    domain = sympy.QQ[sympy.I]
    solsX1 = F0.subs({x:1}).all_roots(multiple=False,radicals=False)
    solsY1 = F0.subs({y:1}).all_roots(multiple=False,radicals=False)

    sols   = [ (1,yi,0) for yi,_ in solsX1 ]
    sols.extend( [ (xi,1,0) for xi,_ in solsY1 ] )

    grad = [F.diff(var) for var in [x,y,_z]]
    for xi,yi,zi in sols:
        if not any( map(lambda e: e.subs({x:xi,y:yi,_z:zi}),grad) ):
            S.append((xi,yi,zi))

    return S


def singular_points(f,x,y):
    """
    Returns the points in P^2_C at which f = f(x,y) is singular.

    Returns a list [..., (xi,yi,zi),...] where P is the location of the
    point in the projective plane P^2_C.
    """
    S = _singular_points_finite(f,x,y)
    S_oo = _singular_points_infinite(f,x,y)
    S.extend(S_oo)

    # obtain the multiplicity, delta invariant, and branching number
    info = _compute_singularity_info(f,x,y,S)

    return zip(S,info)



def _compute_singularity_info_finte(f,x,y,singular_pt):
    """
    Returns the singularity info in the finite case
    """
    return 0


def _compute_singularity_info_infinite(f,x,y,singular_pt):
    """    
    Returns the singularity info in the finite case
    """
    return 0


def _compute_singularity_info(f,x,y,singular_pts):
    """
    For each singularity [alpha, beta, gamma], compute the
    information [m, delta, r] where

    * m = multiplicity
    * delta = delta invariant
    * r = branching number.
    """
    info = []
    
    def puiseux_filter(P,beta):
        if P[1].subs({_t:0}) == beta:
            return True
        return False
    
    F,_ = homogenize(f,x,y,_z)
    for alpha, beta, gamma in singular_pts:
        # compute the Puiseux series at the projective point [alpha,
        # beta, gamma]. If on the line at infinity, make the
        # appropriate variable transformation.
        if gamma:
            # finite case
            P = puiseux(f,x,y,alpha,nterms=1,parametric=_t)
            P_beta = filter(lambda p: puiseux_filter(p,beta), P)
            
            # build non-parametric versions, too
            P_beta_x = []
            for X,Y in P_beta:
                solns = sympy.solve(x-X,_t)
                for TT in solns:
                    P_beta_x.append(Y.subs(_t,TT))

        else:
            # infinite case: z=0. Make the appropriate transformation. 
            # puiseux(...) seems to be faster when expanding about zero
            # than anything else so we first check if any of the other 
            # coordinates are zero.
#            pdb.set_trace()
            
            if beta: 
                var = x; var0 = alpha
                g = F.subs(y,beta)
            else:
                var = y; var0 = beta
                g = F.subs(x,alpha)
            
            alpha = var0
            beta  = 0
            P = puiseux(g,var,_z,alpha,nterms=1,parametric=_t)
            P_beta = filter(lambda p: puiseux_filter(p,beta), P)
            

        # multiplicity
        m = 0
        for X,Y in P_beta:
            X = X - alpha
            Y = Y - beta
            ri = abs( X.leadterm(_t)[1] )
            si = abs( Y.leadterm(_t)[1] )
            m += min(ri,si)

        # branching number
        r = len(P_beta)
        
        # delta invariant
        delta = 0
        for j in range(len(P_beta_x)):
            rj     = r
            IntPj  = Int(j,P_beta_x,x)
            delta += sympy.Rational(rj * IntPj - rj + 1, 2)

        info.append( (m,delta,r) )

    return info

                

   
if __name__ == '__main__':
    print '=== Module Test: singularities.py ==='
    from sympy.abc import x,y
    
    f1 = (x**2 - x + 1)*y**2 - 2*x**2*y + x**4
    f2 = y**3 + 2*x**3*y - x**7
    f3 = (y**2-x**2)*(x-1)*(2*x-3) - 4*(x**2+y**2-2*x)**2
    f4 = y**2 + x**3 - x**2
    f5 = (x**2 + y**2)**3 + 3*x**2*y - y**3
    f6 = y**4 - y**2*x + x**2
    f7 = y**3 - (x**3 + y)**2 + 1
    f8 = x**2*y**6 + 2*x**3*y**5 - 1
    f9 = 2*x**7*y + 2*x**7 + y**3 + 3*y**2 + 3*y
    f10= (x**3)*y**4 + 4*x**2*y**2 + 2*x**3*y - 1
    

    fs = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10]

    print '\nSingular points of curves:'
    for f in fs:
        print '\n\tCurve:'
        sympy.pprint(f)
        print '\nall singular points:'
        singular_pts = singular_points(f,x,y)
#        sympy.pprint(singular_pts)
        for singular_pt in singular_pts:
            print "Point:"
            sympy.pprint(singular_pt[0])
            sympy.pprint(singular_pt[1])
