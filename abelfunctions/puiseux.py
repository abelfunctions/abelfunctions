"""Puiseux Series :mod:`abelfunctions.puiseux`
===========================================

Tools for computing Puiseux series. A necessary component for computing
integral bases and with Riemann surfaces.

Classes
-------

.. autosummary::

    PuiseuxTSeries
    PuiseuxXSeries

Functions
---------

.. autosummary::

    puiseux

References
----------

.. [Duval] D. Duval, "Rational puiseux expansions", Compositio
   Mathematica, vol. 70, no. 2, pp. 119-154, 1989.

Examples
--------

Contents
--------

"""

import numpy
import sympy

from operator import itemgetter
from sympy.core.numbers import Zero
from utilities import cached_function

# we use global symbols for Sympy caching performance
_Z = sympy.Dummy('Z')


def _coefficient(F):
    """Returns a dict of coefficients of ``F`` indexed by monomial powers.

    Parameters
    ----------
    F : sympy.Expr or sympy.Poly

    Returns
    -------
    dict
        A dictionary with keys ``(i,j)`` and values :math:`a_{ij}` where
        each monomial of ``F`` is of the form :math:`a_{ij} X^j Y^i`.

    Examples
    --------

        >>> from sympy impport Poly
        >>> from sympy.abc import x,y
        >>> f = Poly(y**2 + x**2*(x+1))
        >>> _coefficient(f) == {(0, 2): -1, (0, 3): -1, (2, 0): 1}
        True

    """
    d = {}
    monoms = F.monoms()
    coeffs = F.coeffs()
    for a,(j,i) in zip(coeffs,monoms):   # lexicographic ordering
        d[(i,j)] = a
    return d


def _bezout(q, m):
    r"""Returns :math:`u,v` such that :math:`uq+mv=1`.

    Parameters
    ----------
    q, m : integer
        Two coprime integers with :math:`q > 0`.

    Returns
    -------
    tuple of integers

    """
    u,v,g = sympy.gcdex(q,m)
    if u*q+v*m != 1: raise ValueError("Bezout algorithm failed.")
    return u,v


def _square_free(Phi, var):
    r"""Returns the square-free factors of the polynomial ``Phi``.

    Returns a list of tuples :math:`(\Psi, r)` such that each
    :math:`\Psi` is square-free, pairwise coprime, and

    .. math::

        \Phi = \prod \Psi^r`.

    Parameters
    ----------
    Phi : sympy.Polynomial
    var : sympy.Symbol
        A polynomial ``Phi`` in the variable ``var``.

    Returns
    -------
    list
        The square-free factors and exponents :math:`(\Psi,r)`.

    Notes
    -----
    Such a decomposition can be obtained with derivations and gcd
    computations without any factorization algorithm. It's implemented
    in sympy as ``sympy.sqf`` and ``sympy.sqf_list``

    """
    return sympy.sqf_list(Phi,var)[1]


def _new_polynomial(F, X, Y, tau, l):
    r""" Computes the next iterate of the Newton-Puiseux algorithms.

    Given the Puiseux data :math:`\tau = (q,\mu,m,\beta,\eta)`
    :meth:`_new_polynomial` returns

    .. math::

        \tilde{F} = F(\mu X^q, X^m(\beta+Y)) / X^l

    In this algorithm, the choice of parameters will always result in

    .. math::

        \tilde{F} \in \mathbb{L}(\mu,\beta)[X,Y]

    .. note::

        Calling sympy.Poly() with new generators takes a long time.
        Hence, the manual technique used in this code.

    Parameters
    ----------
    F : sympy.Expr
    X,Y : sympy.Symbol
        ``F`` is a polynomial in the variables ``X`` and ``Y``.
    tau : tuple
        The Puiseux data from which the new polynomial and term in the
        Puiseux series is calculated.
    l : integer

    Returns
    -------
    sympy.Expr
        The transformed polynomial :math:`\tilde{F}`.

    """
    q,mu,m,beta,eta = tau
    d = {}

    # for each monomial of the form
    #
    #     c * x**a * y**b
    #
    # compute the appropriate new terms after applying the
    # transformation
    #
    #     x |--> mu * x**q
    #     y |--> eta * x**m * (beta+y)
    #
    for (a,b),c in F.as_dict().iteritems():
        binom = sympy.binomial_coefficients_list(b)
        new_a = int(q*a + m*b)
        for i in xrange(b+1):
            # the coefficient of the x***(qa+mb) * y**i term
            new_c = c * (mu**a) * (eta**b) * (binom[i]) * (beta**(b-i))
            try:
                d[(new_a,i)] += new_c
            except KeyError:
                d[(new_a,i)] = new_c

    # now perform the polynomial division by x**l. In the case when the
    # curve is singular there will be a cancellation resulting in a term
    # of the form (0,0):0 . Creating a new polynomial containing such a
    # term will result in the zero polynomial.
    new_d = dict([((a-l,b),c) for (a,b),c in d.iteritems()])
    Fnew = sympy.Poly.from_dict(new_d, gens=[X,Y], domain=sympy.EX)
    return Fnew


def polygon(F, X, Y, I):
    r"""Returns the Newton polygon data corresponding to ``F``.

    If ``I=2`` the correspondence is only with the segments with
    negative slope.

    The segment :math:`\Delta` corresponding to the list
    :math:`(q,m,l,\Phi)` is on the line :math:`qj + mi = l` in the
    :math:`(i,j)`-plane and where

    .. math::

        \Phi = \sum_{(i,j) \in \Delta} a_{ij} Z^{(i-i_0)/q}

    where :math:`i_0` is the smallest value of :math:`i` such that there
    is a point :math:`(i,j) \in \Delta`. Note that :math:`\Phi \in
    \mathbb{L}[Z]`.

    Parameters
    ----------
    F : Sympy Poly
        A polynomial in :math:`\mathbb{L}[X,Y]`.
    X,Y : Sympy Symbol
        The dependent and independent variables, respectively.
    I : int

    Returns
    -------
    list
        A list of tuples ``(q,m,l,\Phi)`` where ``q,m,l`` are integers with
        :math:`(q,m) = 1`, :math:`q>0`, and :math:`\Phi \in
        \mathbb{L}[Z]`.

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
        hull_with_bdry = [p for p in support
                          if not hull.encloses(sympy.Point(p))]
    newton = []

    # find the start and end points (0,J) and (I,0). Include points
    # along the i-axis if I==1. Otherwise, only include points with
    # negative slope if I==2
    vertices = hull.points if isinstance(hull,sympy.Segment) else hull.vertices
    JJ = min(pt.y for pt in vertices if pt.x == 0)
    if I == 2: II = min(pt.x for pt in vertices if pt.y == 0)
    else:      II = max(pt.x for pt in vertices if pt.y == 0)
    testslope = -float(JJ)/II

    # determine largest slope with (0,JJ). If this is greater than the
    # test slope then there exist points above the line connecting
    # (0,JJ) with (II,0) this fact is used to deal with certain
    # borderline cases
    include_borderline_colinear = (type(hull) == sympy.Segment)
    for (i,j) in hull_with_bdry:
        # when the point is on the j-axis, the onle one we want is the
        # point (0,JJ)
        if i == 0:
            if j == JJ: slope = -sympy.oo
            else:       slope = sympy.oo
        else:           slope = float(j - JJ)/i

        if slope > testslope:
            include_borderline_colinear = True
            break

    # loop through all points on the boundary and determine if it's in
    # the newton polygon using a testslope method
    for (i,j) in hull_with_bdry:
        # when the point is on the j-axis, the onle one we want is the
        # point (0,JJ)
        if i == 0:
            if j == JJ: slope = -sympy.oo
            else:       slope = sympy.oo
        else:           slope = float(j - JJ)/i

        # if the slope is less than the test slope or if we're in the
        # case where we include points along the j=0 line then add the
        # point to the newton polygon
        if (slope < testslope) or (I==1 and j==0):
            newton.append((i,j))
        elif (slope == testslope) and include_borderline_colinear:
            # borderline case is when there is only one segment from
            # (0,JJ) to (II,0). When this is the case, include all
            # points whose slope matches testslope
            newton.append((i,j))

    newton.sort(key=itemgetter(1),reverse=True) # sort second in j-th coord
    newton.sort(key=itemgetter(0))              # sort first in i-th coord


    # now that we have the newton polygon we compute the parameters
    # (q,m,l,Phi) for each side Delta on the polygon.
    params = []
    eps = 1e-14
    N = len(newton)
    n = 0
    while n < N-1:
        # determine slope of current line
        side = [newton[n],newton[n+1]]
        sideslope = sympy.Rational(side[1][1] - side[0][1],
                                   side[1][0] - side[0][0])

        # check against all following points for colinearity by
        # comparing slopes. append all colinear points to side
        k = 1
        for pt in newton[n+2:]:
            slope = float(pt[1] - side[0][1])/(pt[0] - side[0][0])
            if abs(slope - sideslope) < eps:
                side.append(pt)
                k += 1
            else:
                # when we reach the end of the newton polygon we need to
                # shift the value of k a little bit so that the last
                # side is correctly captured if k == N-n-1: k -= 1
                break
        n += k

        # compute q,m,l such that qj + mi = l and Phi
        q   = sideslope.q
        m   = -sideslope.p
        l   = q*side[0][1] + m*side[0][0]
        i0  = min(side, key=itemgetter(0))[0]
        Phi = sum(a[(i,j)]*_Z**sympy.Rational(i-i0,q) for (i,j) in side)
        Phi = sympy.Poly(Phi,_Z)
        params.append((q,m,l,Phi))

    return params


def singular(F, X, Y, pi=None):
    r"""Returns the singular term data of all Puiseux series above some
    :math:`x=\alpha`.

    Computes a collection of pairs :math:`(\pi_i, F_i)` where
    :math:`\pi_i` is a collection of :math:`\tau = (q,\mu,m,\beta,\eta)`
    data representing the "singular parts" of all of the Puiseux series
    expansions above some :math:`x=\alpha`.

    Parameters
    ----------
    F : sympy.Expr
    X,Y : sympy.Symbol
        ``F`` is a polynomial in the variables ``X`` and ``Y``.
    pi : list
        (Default: ``None``) a list of :math:`\tau` data. When provided
        it implies that additional singular terms need to be computed.
        (Note that this algorithm can be recursive.)

    Returns
    -------
    list
        List of pairs :math:`(\pi,F)` each representing the singular
        part of a :class:`PuiseuxTSeries`.

    """
    S = []

    # set a flag for whether or not this is a new Puiseux t-series we're
    # computing or if we need to compute more terms of an existing
    # t-series dataset.
    I = 2 if pi else 1

    for (tau,l,r) in singular_term(F,X,Y,I):
        pi1 = pi + [tau]
        F1 = _new_polynomial(F,X,Y,tau,l)

        # if r == 1 then we've determined the singular part of the
        # current puiseux t-series. otherwise, we need to compute more
        # singular terms
        if r == 1:
            S.append((pi1,F1))
        else:
            S.extend(singular(F1,X,Y,pi1))

    return S


def singular_term(F, X, Y, I):
    """Computes a single set of singular terms of the Puiseux expansions.

    For :math:`I=1`, the function computes the first term of each finite
    Puiseux series expansion. For :math:`I=2`, it computes the other
    singular terms of the expansion, if necessary.

    Parameters
    ----------
    F : sympy.Expr
    X,Y : sympy.Symbol
        A polynomial :math:`F = F(X,Y)`. At first the curve itself but
        later an intermediate polynomial from which further terms are
        computed.
    I : int
        A flag indicating whether or not to compute the first (``I=1``)
        or subsequent (``I=2``) singular terms.

    Returns
    -------
    list
        A list of :math:`\tau` data representing the singular part of a
        Puiseux series.

    """
    T = []

    # if the curve is singular then compute the singular tuples and
    # return.  otherwise, use the standard newton polygon method
    if is_singular_curve(F, X, Y):
        for (q,m,l,Phi) in desingularize(F, X, Y):
            for eta in Phi.all_roots(radicals=True):
                eta = sympy.together(eta)
                tau = (q,1,m,1,eta)
                T.append((tau,0,1))
        return T

    # for each side of the newton polygon
    for (q,m,l,Phi) in polygon(F, X, Y, I):
        u,v = _bezout(q,m)

        # each newton polygon side has a characteristic polynomial. For
        # each square-free factor, each root corresponds to a K-term
        for (Psi,r) in _square_free(Phi, _Z):
            # compute the roots of Psi. Use the RootOf construct if
            # possible. In the case when Psi is over EX (i.e. when
            # RootOf doesn't work) then compute symbolic roots.
            Psi = sympy.Poly(Psi, _Z)
            try:
                roots = Psi.all_roots(radicals=True)
            except NotImplementedError:
                roots = sympy.roots(Psi,_Z).keys()

            # for each root, compute the K-term consisting of a tau
            # term, the largest x*l factor of the polynomial F, and a
            # flag, r, indicating whether or not we're done determining
            # the singular term
            for xi in roots:
                mu = xi**(-v)
                beta = sympy.together(xi**u)
                tau = (q,mu,m,beta,1)
                T.append((tau,l,r))
    return T

def is_singular_curve(f,x,y):
    r"""Returns ``True`` if :math:`f = f(x,y)` is singular at :math:`x=0`.

    Parameters
    ----------
    f : sympy.Expr
    x,y : sympy.Symbol
        A complex plane algebraic curve :math:`f = f(x,y)`.

    Returns
    -------
    boolean

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


def desingularize(f, x, y):
    r"""Returns Puiseux expansion data initializing a singular Puiseux series.

    Parameters
    ----------
    f : sympy.Expr
    x,y : sympy.Symbol
        A complex plane algebraic curve :math:`f = f(x,y)` which is
        singular (:math:`y = \infty`) at :math:`x = 0`.

    Returns
    -------
    list
        A single :math:`\tau` term initalizing the Puiseux series at
        this :math:`x = 0`.

    """
    coeffs = _coefficient(f)
    c = coeffs.pop((0,0))

    # for each monomial c x**j y**i find the dominant term: that is the
    # one that cancels out the constant term c. This is done by
    # substituting x = T**q, y = eta T**m giving the monomial c eta**i *
    # T**(qj + mi). To balance the equation (kill the constant term) we
    # need qj + mi = 0. q = i, m = -j satisfies this equation.
    #
    # Finally, we need to check that this choice of q,m doesn't
    # introduce terms with negative exponent in the curve.
    q,m = (1,1)
    for (i,j),aij in coeffs.iteritems():
        g = sympy.gcd(i,j)
        q = sympy.Rational(i,g)
        m = -sympy.Rational(j,g)

        # check if the other terms remain positive.
        if all(q*jj + m*ii>=0 for ii,jj in coeffs.keys()):
            break

    if (q,m) == (1,1):
        raise ValueError("Unable to compute singular term.")

    # now compute the values of eta that cancel the constant term c
    Phi = [aij*_Z**sympy.Rational(i,q) for (i,j) in coeffs.keys()
           if q*j+m*i == 0]
    Phi = sympy.Poly(sum(Phi) + c, _Z)
    return [(q,m,0,Phi)]


def build_qh_dict(q):
    """
    Given an array, q, of length R efficientily compute

        q_{h_1}^{h_2} = \prod_{k=h_1+1}^{h_2} q[k-1], 0 \leq h1 \leq h2 \leq R

    values used int "Rational pusieux expansions" by D. Duval. Returns
    a dictionary with keys (h1,h2).
    """
    R = len(q)
    qh = dict(((h,h),1) for h in xrange(R+1))

    # for each q[h], determine which existing entries in the qh dict
    # we can append to to create new entries.
    for h in xrange(R):
        for (h1,h2), qh1h2 in qh.items():
            # indexing shift.
            if h2 == h:
                qh[(h1,h2+1)] = qh1h2*q[h]

    return qh


def build_series(pi):
    r"""Computes the x- and y-part terms from a singular :math:`\pi`.

    The x- and y-parts of a PuiseuxTSeries are computed using repeated
    applications of the map

    ..math::

        \tau_i = (q, \mu, m, \beta, \eta),
        X \mapsto \mu X^q
        Y \mapsto X^m(\beta + \eta Y)

    where :math:`\tau_i` is the coefficient data of a singular term of
    the Puiseux series as computed by the functions :func:`singular`.

    Parameters
    ----------
    pi : list
        A list of :math:`\tau` terms representing the singular part of a
        PuiseuxTSeries.

    Returns
    -------
    e, lam, terms
        The ramification index :math:`e`, the coefficient
        :math:`\lambda` of the x-part, and a dictionary of terms
        representing the y-part whose keys are the exponents of the
        terms and values are the coefficients.

    Notes
    -----

    Uses the formulas from p. 135,136 of "Rational puiseux expansions"
    by D. Duval with the change that indexing of m,beta,mu,eta are all
    shifted down by one. The indexing on qh matches the notation of
    D. Duval.

    """
    q,mu,m,beta,eta = zip(*pi)
    R = len(pi)
    qh = build_qh_dict(q)

    # compute the parameters appearing in the x-part
    e = qh[(0,R)]
    lam = reduce(lambda z1,z2: z1*z2,
                 (mu[k]**qh[(0,k)] for k in xrange(R)))

    # compute the terms of the y-part
    terms = {}
    n_h = sympy.S(0)
    eta_h = sympy.S(1)
    for h in range(0,R):
        # compute a term exponent: n_h
        eta_h *= eta[h]
        n_h += m[h] * qh[(h+1,R)]

        # compute a term coefficient: alpha_h
        alpha_h = sympy.S(1)
        for i in range(0,h):
            alpha_h *= mu[i+1]**sum(m[j]*qh[(j+1,i)] for j in xrange(0,i))
        for i in range(h,R-1):
            alpha_h *= mu[i+1]**sum(m[j]*qh[(j+1,i)] for j in xrange(0,h))
        alpha_h *= eta_h*beta[h]

        # add to the terms dictionary
        terms[n_h] = alpha_h

    return e, lam, terms.items()


def puiseux(f, x, y, alpha, beta=None, t=sympy.Symbol('t'), parametric=False,
            order=None, nterms=None):
    r"""Returns a list of Puiseux series lying above :math:`x=\alpha`.

    Computes a list of :class:`PuiseuxTSeries` objects representing the
    Puiseux series at the lift of :math:`x=\alpha`. Each of these series
    can be further decomposed into :class:`PuiseuxXSeries` objects.

    Parameters
    ----------
    f : sympy.Expr
    x : sympy.Symbol
    y : sympy.Symbol
    alpha : complex
        The x-point :math:`x=\alpha` at which to compute the set of
        Puiseux series. `sympy.oo` is an allowed input
    beta : complex
        (Default: `None`) If provided, only returns the parametric
        Puiseux series centered at :math:`(x,y) = (\alpha,\beta)`
    var : sympy.Symbol
        (Default: `sympy.Symbol('t')`) The variable in which the
        parametric Puiseux series expansions are given.
    parametric : bool
        (Default: `False`) If `True`, returns a list of
        `PuiseuxTSeries`. Otherwise, returns a list of `PuiseuxXSeries`.
    order : int
        (Default: `None`) If provided, returns Puiseux series expansions
        up the the specified order.
    nterms : int
        (Default: `None`) If provided, returns Puiseux series expansions
        with the specified number of terms.

    Returns
    -------
    list
        A list of :class:`PuiseuxTSeries` objects.

    """
    # scale and shift f accordingly: the algorithms for computing the
    # Puiseux series terms are centered at x=0
    if alpha == sympy.oo:
        f = (f.subs(x, 1/x) * x**(f.degree(x)))
    else:
        f = f.subs(x, x + alpha)
    f = sympy.Poly(f,[x,y])

    # compute the puiseux series data and construct the corresponding
    # PuiseuxTSeries objects.
    singular_data = singular(f,x,y,[])
    series = [PuiseuxTSeries(f, x, y, alpha, datum, t=t)
              for datum in singular_data]
    for P in series:
        P.extend(order=order, nterms=nterms)

    # if y-value, beta, is given then return only the puiseux series
    # such that Py(0) = beta
    if beta is not None:
        series = [Pi for Pi in series if Pi.eval_y(0) == beta]

    # compute all x-series if requested
    if not parametric:
        series = [px for P in series for px in P.xseries()]
    return series


class PuiseuxTSeries(object):
    r"""A Puiseux t-series about some place :math:`(\alpha, \beta) \in X`.

    A parametric Puiseux series :math:`P(t)` centered at :math:`(x,y) =
    (\alpha, \beta)` is given in terms of a pair of functions

    .. math::

        x(t) = \alpha + \lambda t^e, \\
        y(t) = \sum_{h=0}^\infty \alpha_h t^{n_h},

    where :math:`x(0) = \alpha, y(0) = \beta`.

    The primary reference for the notation and computational method of
    these Puiseux series is D. Duval.


    Attributes
    ----------
    f : sympy.Expr
    x : sympy.Symbol
    y : sympy.Symbol
    x0 : complex
        The x-center of the Puiseux series expansion.
    ramification_index : sympy.Rational
        The ramification index :math:`e`.
    terms : list
        A list of exponent-coefficient pairs representing the y-series.
    order : int
        The order of the Puiseux series expansion.

    Methods
    -------
    xseries
    extend
    eval_x
    eval_y
    evalf_x
    evalf_y

    """
    def __init__(self, f, x, y, x0, singular_data, t=sympy.Symbol('t'),
                 order=None):
        r"""Initialize a PuiseuxTSeries using a set of :math:`\pi = \{\tau\}`
        data.

        Parameters
        ----------
        f : sympy.Expr
        x : sympy.Symbol
        y : sympy.Symbol
        x0 : complex
            The x-center of the Puiseux series expansion.
        singular_data : list
            The output of :func:`singular`.

        """
        self.f = f
        self.x = x
        self.y = y
        self.t = t
        self.x0 = x0

        pi, F = singular_data
        ramification_index, lamb, terms = build_series(pi)
        self._lamb = lamb
        self._pi = pi
        self._F = F
        self.ramification_index = ramification_index
        self.terms = terms
        self.order = order

    def __repr__(self):
        """Print the x- and y-parts of the Puiseux series."""
        # print the x-part
        s = '%s(%s) = '%(self.x,self.t)
        s += '%s + '%(self.x0) if self.x0 != 0 else ''
        s += '(%s)%s**%s\n'%(self._lamb, self.t, self.ramification_index)

        # print the y-part
        s += '%s(%s) = '%(self.y, self.t)
        ss = '%s'%self.t
        for exp,coeff in self.terms:
            s += ' + (%s)'%coeff
            if exp != sympy.S(0):
                s += ss + '**%s'%exp

        s += '+ O(%s**%s)'%(self.t, self.order)
        return s

    def xseries(self, all_conjugates=True):
        r"""Returns the corresponding x-series.

        Parameters
        ----------
        all_conjugates : bool
            (default: True) If ``True``, returns all conjugates
            x-representations of this Puiseux t-series. If ``False``,
            only returns one representative.

        Returns
        -------
        list
            List of PuiseuxXSeries representations of this PuiseuxTSeries.

        """
        e = self.ramification_index
        lamb = self._lamb
        xseries = []
        order = self.order

        # compute the e-th roots of lambda
        C = sympy.root(1/lamb,e)
        if all_conjugates:
            conjugates = [C*sympy.exp(2*sympy.pi*sympy.I*sympy.Rational(k,e))
                          for k in range(e)]
        else:
            conjugates = [C]

        # compute each conjugate x-series
        for c in conjugates:
            terms = [(nh/e, alphah*c**nh) for nh,alphah in self.terms]
            p = PuiseuxXSeries(self.f, self.x, self.y, self.x0, terms,
                               order=order, ramification_index=e)
            xseries.append(p)
        return xseries

    def nterms(self):
        """Returns the number of non-zero computed terms.

        Parameters
        ----------
        None

        Returns
        -------
        int

        """
        return len(self._pi)

    def add_term(self):
        r"""Add the next tau term to this Puiseux t-series in-place.

        .. note::

            Calls :meth:`_update_terms` as a subroutine in order to
            actually compute / recompute the next term in the terms
            dictionary and :meth:`_update_xseries` to update the child
            x-series associated with this t-series.

        Parameters
        ----------
        None

        Returns
        -------
        boolean
            Returns ``False`` if a finite Puiseux expansion is encountered.

        .. note::

            The calculation of the next :math:`\alpha` term involves a
            division that doesn't need to be there but avoiding this
            division requires some more state. The best way to hand this
            state is something that needs to be thought of.

        """
        # extract the tau data and get the current intermediate
        # polynomial F
        q,mu,m,beta,eta = self._pi[-1]
        F = self._F
        x = self.x
        y = self.y

        # get the set of all exponents j where a_{0,j} x^j is a term in the
        # polynomial F. if this set is empty when we've found a finite
        # puiseux expansion.
        a = dict(F.terms())
        ms = [j for (j,i) in a.keys() if i==0 and j!=0]
        if ms == []:
            return False

        # compute the next regular tau term and polynomial: m is equal to
        # the smallest degree x^m term and beta is the raito of the
        # coefficient of this term with the coefficient of y.
        m_next = min(ms)
        beta_next = sympy.together(-a[(m_next,0)]/a[(0,1)])
        tau = (1, 1, m_next, beta_next, 1)
        F = _new_polynomial(F, x, y, tau, m_next)

        # update the internal state, including the computation of the
        # actual y-series term and exponent
        self._pi.append(tau)
        self._F = F

        # compute the next term of the y-series. the formula for doing
        # so is greatly simplified when tau is regular
        n_prev, alpha_prev = max(self.terms)
        alpha = beta_next * alpha_prev / beta
        n = n_prev + m_next
        self.terms.append((n,alpha))

        # update the order of the expansion
        self.order = n+1
        return True

    def extend(self, order=None, nterms=None):
        """Extends the series in place.

        Computes additional terms in the Puiseux series up to the
        specified `order` or with `nterms` number of non-zero terms. If
        neither `degree` nor `nterms` are provided then the next
        non-zero term will be added to this t-series.

        Parameters
        ----------
        order : int, optional
        nterms : int, optional
            The desired degree or number of terms to extend the series to.

        Returns
        -------
        None

        """
        # if neither are provided then increment nterms
        if not order and not nterms:
            nterms = self.nterms() + 1

        # build an appropriate comparison function `cmp` and bound for
        # determining if we're done extending this series
        if order:
            # return the degree of the last term
            cmp = lambda terms: max(terms)[0]
            bound = order
        elif nterms:
            # return the length of terms
            cmp = lambda terms: len(terms)
            bound = nterms

        # main loop: keep adding terms until the stopping condition is
        # satisfied
        while cmp(self.terms) < bound:
            if not self.add_term():
                # encountered a finite Puiseux expansion
                break

    def eval_x(self, t):
        r"""Symbolic evaluation of the x-part of the Puiseux series.

        Parameters
        ----------
        t : sympy.Expr

        Returns
        -------
        complex

        """
        xval = self.x0 + self._lamb*t**self._e
        return xval

    def evalf_x(self, t):
        r"""Numerical evaluation of the x-part of the Puiseux series.

        Parameters
        ----------
        t : complex

        Returns
        -------
        complex

        """
        xval = self.eval_x(t)
        return numpy.complex(xval.n())

    def eval_y(self, t):
        r"""Symbolic evaluation of the y-part of the Puiseux series.

        Parameters
        ----------
        t : complex

        Returns
        -------
        complex

        Notes
        -----

        This can be sped up using a Holder-like fast exponent evaluation
        trick.

        """
        yval = sympy.S(0)
        for n, alpha in self.terms:
            yval += alpha * t**n
        return yval

    def evalf_y(self, t):
        r"""Numerical evaluation of the y-part of the Puiseux series.

        Parameters
        ----------
        t : complex

        Returns
        -------
        complex

        Notes
        -----

        This can be sped up using a Holder-like fast exponent evaluation
        trick.

        """
        yval = self.eval_y(t)
        return numpy.complex(yval.n())


class PuiseuxXSeries(object):
    r"""A Puiseux x-series centered at :math:`x = x_0`.

    A Puiseux series :math:`p(x)` centered at :math:`x = x_0` is a
    series of the form

    .. math::

        p(x) = \sum_{h=0}^\infty \alpha_h (x - x_0)^{n_h/e},

    The primary reference for the notation and computational method of
    these Puiseux series is D. Duval.

    Attributes
    ----------
    f : sympy.Expr
    x : sympy.Symbol
    y : sympy.Symbol
    x0 : complex
        The center of the Puiseux series expansion.
    terms : tuple
        A list of exponent-coefficient pairs.
    order : sympy.Rational
        The order of the Puiseux series expansion.
    ramification_index : sympy.Rational
        The ramification index :math:`e`.

    Methods
    -------
    eval
    evalf
    valuation
    as_sympy_expr

    """
    def __init__(self, f, x, y, x0, obj, order=None, ramification_index=None):
        r"""Initialize a PuiseuxXSeries.

        A PuiseuxXSeries is initialized from a RiemannSurface, center
        ``x0``, and choice of data:

        * list, tuple: an iterable of (exponent, coefficient) 2-tuples
        * dictionary: with exponents as keys and coefficients for values
        * sympy.Expr: computed as a series representation in
          :math:`x-x_0`.

        Optionally, the order and ramification index of the series can
        be directly specified. Otherwise, they are guessed from the
        given data.

        Parameters
        ----------
        f : sympy.Expr
        x : sympy.Symbol
        y : sympy.Symbol
        x0 : complex
            The center of the Puiseux series expansion.
        obj : list, tuple, dict, or sympy.Expr
            Data from which the series is initialized. (See
            documentation above.)
        order : sympy.Rational, optional
            The order of the Puiseux series. Truncates the given data,
            if necessary. If `obj` is a sympy.Expr then computes the
            series expansion of obj in :math:`x-x_0`.
        ramification_index : int, optional
            If not provided, the ramification index will be set to the
            gcd of the denominators appearing in the term exponents.

        """
        self.f = f
        self.x = x
        self.y = y
        self.x0 = x0

        # initialize properties
        self._terms = None
        self._order = None
        self.ramification_index = None

        # intitalize terms from given object
        terms = self.initialize_terms(obj, order=order)
        self.terms = terms #tuple(sorted(terms, key=itemgetter(0)))

        # determine ramification index of this Puiseux series. if not
        # explicitly given in construction it is assumed from the
        # provided exponents and order
        if not ramification_index:
            exp, _ = map(list, zip(*self.terms))
            exp += [self.order]
            denoms = map(sympy.denom, exp)
            ramification_index = sympy.gcd(denoms)
        self.ramification_index = ramification_index
        self.order = order
        self._hash = None

    ##################
    # property: order
    ##################
    def get_order(self):
        return self._order
    def set_order(self, order):
        # if order isn't specified then set order equal to the exponent
        # of the largest non-zero term plus 1/ramification_index
        if order:
            self._order = order
        else:
            self._order = max(exp for exp,coeff in self.terms) + \
                          sympy.Rational(1,self.ramification_index)

        # truncate if new order is less than previous
        self.terms = tuple((exp,coeff) for exp,coeff in self.terms
                           if exp < self._order)
    order = property(get_order, set_order)

    ##################
    # property: terms
    ##################
    def get_terms(self):
        return self._terms
    def set_terms(self, terms):
        # filter out zero terms unless the series is the zero
        # series. (useful for accumulation.)
        terms = tuple(sorted(
            [(exp,coeff) for exp,coeff in terms if coeff != 0],
            key=itemgetter(0)))

        if not terms:
            terms = ((0,0),)
        self._terms = terms

        # if order isn't set then assume all known terms are given. if
        # order ends up being zero then set to infinity
        if not self._order:
            order = max(exp for exp,coeff in self._terms)
            self._order = order if order else sympy.oo
    terms = property(get_terms, set_terms)


    def __hash__(self):
        """Returns the hash of this PuiseuxXSeries.

        The has of a series is a hash of the Riemann surface, Puiseux
        series expansion center, terms, and order. This is necessary for
        memoization of PuiseuxXSeries. Particularly, in
        :meth:`abelfucntions.integralbasis.integral_basis`.

        """
        if not self._hash:
            self._hash = hash((self.x0,self.terms,self.order))
        return self._hash


    def __repr__(self):
        s = ''
        ss = '%s'%self.x if self.x0 == 0 else '(%s)'%(self.x - self.x0)
        for exp,coeff in self.terms:
            # print the coefficient
            s += ' + (%s)'%coeff
            if exp != 0:
                s += ss + '**%s'%exp

        s += ' + O(%s**%s)'%(ss, self.order)
        return s[3:]


    def initialize_terms(self, obj, order=None):
        """Initialize the terms of the Puiseux series from the object ``obj``.

        A PuiseuxXSeries can be initialized form a tuple, list,
        dictionary, or Sympy ``Expression`` object. See
        :meth:`__init__()` for more information.

        If `obj` is a tuple or list each element is a 2-tuple whose
        first element is the exponent and second element is the
        coefficient of that term. If `obj` is a dictionary then the keys
        are the exponents and the values are the coefficients. If `obj`
        is a Sympy `Expression` then the powers and coefficients are
        determined using `sympy.lseries`.

        Parameters
        ----------
        obj : tuple, list, dict, or Sympify-able expression
        order : int, optional

        Returns
        -------
        tuple of two-tuples

        """
        if isinstance(obj, tuple):
            return obj
        elif isinstance(obj, list):
            return tuple(obj)
        elif isinstance(obj, dict):
            return obj.items()
        else:
            try:
                return self._terms_from_sympy_expression(obj, order)
            except sympy.SympifyError:
                raise ValueError('Cannot initalize PuiseuxXSeries from %s'%obj)

    def _terms_from_sympy_expression(self, expr, order=None):
        r"""Compute series terms in (x-alpha) of a sympy expression.

        .. note::

            Sympy's `series()` function behaves differently depending on
            whether `n` is specified. If `n` is given then the series is
            indeed in :math:`(x-a)` but is written in :math:`x` so no
            shifting is necessary. However, without `n` each term is
            given as a power of :math:`(x-a)` so shifting (or alternate
            parsing) is needed. Very weird.

        Parameters
        ----------
        expr : sympy.Expr
        order : int
            (Default: 5) The desired order of the series approximation
            of `expr`.

        Returns
        -------
        tuple of 2-tuples

        """
        # optimize for various kinds of expressions
        order = order if order else 5
        numer,denom = expr.as_numer_denom()
        if self.x not in expr:
            return self._terms_from_sympy_const(expr,order)
        if numer.is_algebraic_expr() and self.x not in denom:
            return self._terms_from_sympy_polynomial(expr,order)
        elif numer.is_algebraic_expr() and denom.is_algebraic_expr():
            return self._terms_from_sympy_rational(expr,order)
        else:
            return self._terms_from_sympy_generic(expr,order)

    def _terms_from_sympy_const(self, expr, order):
        r"""Returns terms of constant expression.
        """
        return ((0,expr),)

    def _terms_from_sympy_polynomial(self, expr, order):
        r"""Returns terms from polynomial expression."""
        expr = expr.subs({self.x:self.x+self.x0}).expand()
        terms = expr.collect(self.x,evaluate=False).items()
        terms = [(exp.as_coeff_exponent(self.x)[1],coeff)
                 for exp,coeff in terms]
        return tuple(terms)

    def _terms_from_sympy_rational(self, expr, order):
        r"""Returns terms from rational expression.

        Common and potentially slow situation. This method first
        separates the numerator and denominator. The denominator is
        written as

        ..math ::

            d(x) = cx^l \tilde{d}(x)

        where :math:`\tilde{d}(0) = 1`. This is so a fast Taylor series
        calculation can be done on :math:`\tilde{d}`. The factor of
        :math:`cx^l` is introduced back into the resulting series.

        This approach is similar to the
        :func:`abelfunctions.differentials.fast_expand` approach using
        in localizing :class:`Differential` objects.

        """
        numer,denom = expr.as_numer_denom()
        numer = numer.subs({self.x:self.x+self.x0}).expand()
        numer = numer.collect(self.x,evaluate=False).items()
        denom = denom.subs({self.x:self.x+self.x0}).expand()
        lead_coeff, lead_exp = denom.leadterm(self.x)
        denom = (denom/(lead_coeff*self.x**lead_exp)).expand()
        denom = denom.collect(self.x,evaluate=False).items()

        # forward solve the coefficient system. note that r[0]
        # (constant coeff of denom) is nonzero by construction
        N = max(int(round(order)),1)
        q = [0]*N
        for n,qn in numer:
            n = n.as_coeff_exponent(self.x)[1]
            if n < N:
                q[n] = qn
        r = [0]*N
        for n,rn in denom:
            n = n.as_coeff_exponent(self.x)[1]
            if n < N:
                r[n] = rn
        s = [0]*N
        for n in range(N):
            known_terms = sum(r[n-k]*s[k] for k in range(n))
            s[n] = (q[n] - known_terms)/r[0]

        terms = [(n-lead_exp,s[n]/lead_coeff) for n in range(N)
                 if s[n] not in [0,sympy.S(0)]]
        return tuple(terms)

    def _term_from_sympy_generic(self, expr, order):
        r"""Returns terms from a generic sympy expression.

        The slowest method. Uses `sympy.lseries`.
        """
        s = sympy.series(expr, self.x, x0=self.x0, n=None)
        for term in s:
            term = term.subs(self.x,self.x+self.x0)
            coeff, exp = term.as_coeff_exponent(self.x)
            terms.append((exp,coeff))
            if exp >= order:
                break
        return tuple(terms)


    ###########################################################################
    # Operator Overloading
    ###########################################################################
    def __eq__(self, other):
        if isinstance(other, PuiseuxXSeries):
            if (self.terms == other.terms and
                self.order == other.order and
                self.x0 == other.x0):
                return True
        return False

    def __neg__(self):
        terms = tuple((exp,-coeff) for exp,coeff in self.terms)
        order = self.order
        ramification_index = self.ramification_index
        return PuiseuxXSeries(self.f, self.x, self.y, self.x0, terms,
                              order=order,
                              ramification_index=ramification_index)

    def __add__(self, other):
        if not isinstance(other, PuiseuxXSeries):
            order = self.order
            ramification_index = self.ramification_index
            other = PuiseuxXSeries(self.f, self.x, self.y, self.x0, other,
                                   order=order,
                                   ramification_index=ramification_index)
        terms = []
        order = min(self.order, other.order)
        ramification_index = sympy.gcd(self.ramification_index,
                                       other.ramification_index)
        self_terms = dict(self.terms)
        other_terms = dict(other.terms)
        for exp in set(self_terms) | set(other_terms):
            if exp <= order:
                self_coeff = self_terms.get(exp,0)
                other_coeff = other_terms.get(exp,0)
                coeff = self_coeff + other_coeff
                terms.append((exp,coeff))
        terms = tuple(terms)
        return PuiseuxXSeries(self.f, self.x, self.y, self.x0, terms,
                              order=order,
                              ramification_index=ramification_index)

    def __sub__(self, other):
        return self.__add__(-other)

    def __mul__(self, other):
        if not isinstance(other, PuiseuxXSeries):
            # the minimal order required for multiplication to be
            # "loss-less" is (m + N - n) where N is the order of self, n
            # is the degree of the leading order term of self, and m is
            # the degree of the leading order term of other
            if not isinstance(other, sympy.Basic):
                other = sympy.sympify(other)
            n = self.valuation()
            N = self.order
            m = other.leadterm(self.x)[1]
            order = m + N - n
            ramification_index = self.ramification_index
            other = PuiseuxXSeries(self.f, self.x, self.y, self.x0, other,
                                   order=order,
                                   ramification_index=ramification_index)

        # the order of the product (an(x-x0)^n + ... + O(x^N)) *
        # (bn(x-x0)^m + ... + O(x^M)) is equal to the min m+N and
        # n+M. Note that the terms are always in sorted order.
        self_lo = self.terms[0][0]
        other_lo = other.terms[0][0]
        order = min(self_lo + other.order, other_lo + self.order)
        ramification_index = sympy.gcd(self.ramification_index,
                                       other.ramification_index)

        # compute the terms up to order
        terms = {}
        for self_exp, self_coeff in self.terms:
            for other_exp, other_coeff in other.terms:
                exp = self_exp + other_exp
                if exp <= order:
                    coeff = self_coeff*other_coeff
                    try:
                        terms[exp] = coeff + terms[exp]
                    except KeyError:
                        terms[exp] = coeff

        return PuiseuxXSeries(self.f, self.x, self.y, self.x0, terms,
                              order=order,
                              ramification_index=ramification_index)

    def __div__(self, other):
        # note: only works when ``other`` is a sympy.Expr
        if not isinstance(other, sympy.Expr):
            raise NotImplementedError('Can only divide by Sympy Expressions.')

        return self.__mul__(1/other)

    def __pow__(self, e):
        zero = sympy.S(0)
        one = sympy.S(1)
        order = self.order
        ramification_index = self.ramification_index

        # use base-two representation of exponent for efficiency
        current_power = self
        val = PuiseuxXSeries(self.f, self.x, self.y, self.x0, ((zero, one),),
                             order=order,
                             ramification_index=ramification_index)
        while e > 0:
            # if the current power of two appears in the binary
            # expansion of then add it to the result
            if e % 2:
                val = val * current_power
            # compute the next power of two
            current_power = current_power * current_power
            e >>= 1

        return val

    def eval(self, x):
        r"""Symbolic evaluation of the Puiseux series.

        Parameters
        ----------
        x : complex

        Returns
        -------
        complex

        """
        val = sympy.S(0)
        for exponent, coefficient in self.terms:
            val += coefficient * x**exponent
        return val

    def evalf(self, x):
        r"""Numerical evaluation of the Puiseux series.

        Parameters
        ----------
        x : complex

        Returns
        -------
        complex

        """
        val = self.eval(x)
        return numpy.complex(val.n())


    def valuation(self):
        r"""Returns the valuation of this Puiseux series.

        The valuation of a Puiseux x-series is degree of the
        lowest-order non-zero term. Specifically, if

        .. math::

            y(x) = \sum_h=0^\infty \alpha_h (x-x_0)^(n_h/e)

        then the valuation of :math:`y` is :math:`n_0/e`.

        Parameters
        ----------
        None

        Returns
        -------
        sympy.Rational

        """
        return min(self.terms)[0]

    def as_sympy_expr(self):
        r"""Returns the Puiseux series as a sympy expression.

        Parameters
        ----------
        None

        Returns
        -------
        sympy.Expr

        """
        expr = sympy.S(0)
        for exp, coeff in self.terms:
            expr += coeff*(self.x - self.x0)**exp
        return expr
