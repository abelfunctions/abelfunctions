r"""Puiseux Series :mod:`abelfunctions.puiseux`
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

.. [Poteaux] A. Poteaux, M. Rybowicz, "Towards a Symbolic-Numeric Method
   to Compute Puiseux Series: The Modular Part", preprint

Examples
--------

Todo
----

* test y**2 - x: unique pair (-t**2, t)

Contents
--------

"""

import numpy
import sympy
import pdb

from sympy import (
    degree, Point, Segment, Poly, poly, Rational, RootOf, gcd, gcdex, LC,
    expand,
    )

from abelfunctions.utilities import rootofsimp

_Z = sympy.Symbol('_Z')


def newton_polygon_exceptional(H,x,y):
    r"""Computes the exceptional Newton polygon of `H`"""
    d = degree(H.eval(x,0), y)
    return [[(0,0),(d,0)]]

def newton_polygon(H,x,y,additional_points=[]):
    r"""Computes the Newton polygon of `H`.

    It's assumed that the first generator of `H` here is the "dependent
    variable". For example, if `H = H(x,y)` and we are aiming to compute
    a `y`-covering of the complex `x`-sphere then each monomial of `H`
    is of the form

    .. math::

        a_{ij} x^j y^i.


    Parameters
    ----------
    H : sympy.Poly
        Polynomial in `x` and `y`.
    x : sympy.Symbol
        Dependent variable.
    y : sympy.Symbol
        Independent variable.

    Returns
    -------
    list
        Returns a list where each element is a list, representing a side
        of the polygon, which in turn contains tuples representing the
        points on the side.
    """
    # because of the way sympy.convex_hull computes the convex hull we
    # need to remove all points of the form (0,j) and (i,0) where j > j0
    # and i > i0, the points on the axes closest to the origin
    H = H.as_poly(y,x)
    support = map(Point, H.monoms()) + additional_points
    i0 = min(P.x for P in support if P.y == 0)
    j0 = min(P.y for P in support if P.x == 0)
    support = filter(lambda P: (P.x <= i0) and (P.y <= j0), support)
    convex_hull = sympy.convex_hull(*support)

    # special treatment when the hull is just a point or a segment
    if isinstance(convex_hull, Point):
        P = (convex_hull.x,convex_hull.y)
        return [[P]]
    elif isinstance(convex_hull, Segment):
        P = convex_hull.p1
        convex_hull = generalized_polygon_side(convex_hull)
        support.remove(P)
        support.append(convex_hull.p1)
        sides = [convex_hull]
    else:
        # recursive call with generalized point if a generalized newton
        # polygon is needed.
        sides = convex_hull.sides
        first_side = generalized_polygon_side(sides[0])
        if first_side != sides[0]:
            P = first_side.p1
            return newton_polygon(H,x,y,additional_points=[P])

    # convert the sides to lists of points
    polygon = []
    for side in sides:
        polygon_side = [P for P in support if P in side]
        polygon_side = sorted(map(lambda P: (P.x,P.y), polygon_side))
        polygon.append(polygon_side)

        # stop the moment we hit the i-axis. despite the filtration at
        # the start of this function we need this condition to prevent
        # returning to the starting point of the newton polygon.
        #
        # (See test_puiseux.TestNewtonPolygon.test_multiple)
        if side.p2.y == 0: break

    return polygon

def generalized_polygon_side(side):
    r"""Returns the generalization of a side on the Newton polygon.

    A generalized Newton polygon is one where every side has slope no
    less than -1.

    Parameters
    ----------
    side : sympy.Segment

    Returns
    -------
    side
    """
    if side.slope < -1:
        p1,p2 = side.points
        p1y = p2.x + p2.y
        side = Segment((0,p1y),p2)
    return side

def bezout(q,m):
    r"""Returns :math:`u,v` such that :math:`uq+mv=1`.

    Parameters
    ----------
    q,m : integer
        Two coprime integers with :math:`q > 0`.

    Returns
    -------
    tuple of integers

    """
    if q == 1:
        return (1,0)
    u,v,g = gcdex(q,-m)
    return (u,v)

def transform_newton_polynomial(H,x,y,q,m,l,xi):
    r"""Recenters a Newton polynomial at a given singular term.

    Given the Puiseux data :math:`x=\mu x^q, y=x^m(\beta+y)` this
    function returns the polynomial

    .. math::

        \tilde{H} = H(\xi^v x^q, x^m(\xi^u+y)) / x^l.

    where :math:`uq+mv=1`.
    """
    u,v = bezout(q,m)
    newx = ((xi**v)*(x**q)).as_poly(x)
    newy = ((x**m)*(xi**u + y)).as_poly(y)
    quo = x**l
    newH = H.as_poly(x).compose(newx).as_poly(y).compose(newy)
    newH = newH.exquo(quo).as_poly(x,y)
    return newH

def newton_data(H,x,y,exceptional=False):
    r"""Determines the "newton data" associated with each side of the polygon.

    For each side :math:`\Delta` of the Newton polygon of `H` we
    associate the data :math:`(q,m,l,`phi)` where

    .. math::

        \Delta: qj + mi = l \\
        \phi_{\Delta}(t) = \sum_{(i,j) \in \Delta} a_{ij} t^{(i-i_0)/q}

    Here, :math:`a_ij x^j y_i` is a term in the polynomial :math:`H` and
    :math:`i_0` is the smallest value of :math:`i` belonging to the
    polygon side :math:`\Delta`.

    Parameters
    ----------
    H : sympy.Poly
        Polynomial in `x` and `y`.

    Returns
    -------
    list
        A list of the tuples :math:`(q,m,l,\phi)`.
    """
    H = H.as_poly(y,x)
    if exceptional:
        newton = newton_polygon_exceptional(H,x,y)
    else:
        newton = newton_polygon(H,x,y)

    # special case when the newton polygon is a point
    if len(newton[0]) == 1:
        return []

    result = []
    for side in newton:
        i0,j0 = side[0]
        i1,j1 = side[1]
        slope = Rational(j1-j0,i1-i0)
        q = slope.q
        m = -slope.p
        l = min(q*j0 + m*i0, q*j1 + m*i1)
        phi = sum(H.coeff_monomial((i,j))*_Z**Rational(i-i0,q) for i,j in side)
        phi = phi.as_poly(_Z)
        result.append((q,m,l,phi))
    return result


def newton_iteration(G,t,y,n):
    r"""Returns a truncated series `y = y(t)` satisfying

    .. math::

        G(t,y(t)) \equiv 0 \bmod{t^r}

    where $r = \ceil{\log_2{n}}$. Based on the algorithm in [XXX].

    Parameters
    ----------
    G : sympy.Poly
        A polynomial in `t` and `y`.
    n : int
        Requested degree of the series expansion.

    Notes
    -----
    This algorithm returns the series up to order :math:`2^r > n`. Any
    choice of order below :math:`2^r` will return the same series.

    """
    if n < 0:
        raise ValueError('Number of terms must be positive. (n=%d'%n)
    elif n == 0:
        return sympy.S(0)

    phi = G.as_poly(y)
    phiprime = phi.diff(y).as_poly(y)

    try:
        pim1 = Poly(t,t)
        alpha = G.subs({t:0,y:0})   # recenter in y to handle constant term
        gim1 = Poly(-alpha,y)
        sim1 = phiprime.compose(gim1).as_poly(t).invert(pim1)
    except sympy.NotInvertible:
        raise ValueError('Newton iteration for computing regular part of '
                         'Puiseux expansion failed. Curve is not regular '
                         'at center.')

    r = sympy.ceiling(sympy.log(n,2))
    for i in range(r):
        gim1,sim1,pim1 = newton_iteration_step(phi,phiprime,gim1,sim1,pim1,t,y)

    return gim1.as_expr()


def newton_iteration_step(phi,phiprime,gim1,sim1,pim1,t,y):
    r"""Perform a single step of the newton iteration algorithm.

    Parameters
    ----------
    phi, phiprime : sympy.Poly
        Equation and its y-derivative.
    gim1, sim1 : sympy.Poly
        Current solution and inverse (conjugate) modulo `pim1`.
    pim1 : sympy.Poly
        The current modulus. That is, `gim1` is the Taylor series
        solution to `phi(t,gim1) = 0` modulo `pim1`.
    t,y : sympy.Symbol
        Dependent and independent variables, respectively.

    Returns
    -------
    gi,si,pi

    """
    pi = pim1 * Poly(t**2,t)
    gi = gim1.as_poly(t) - phi.compose(gim1).as_poly(t)*sim1
    gi = (gi % pi).as_poly(y)
    si = 2*sim1 - phiprime.compose(gi).as_poly(t)*sim1**2
    si = (si % pi)
    return gi,si,pi


def puiseux_rational(H,x,y,recurse=False):
    r"""Puiseux data for the curve :math:`H` above :math:`(x,y)=(0,0)`.

    Given a polynomial :math:`H = H(x,y)` :func:`puiseux_rational`
    returns the singular parts of all of the Puiseux series centered at
    :math:`x=0, y=0`.

    Parameters
    ----------
    H : sympy.Poly
        A plane curve in `x` and `y`.
    recurse : boolean
        (Default: `True`) A flag used internally to keep track of which
        term in the singular expansion is being computed.

    Returns
    -------
    list of `(G,P,Q)`
        List of tuples where `P` and `Q` are the x- and y-parts of the
        Puiseux series, respectively, and `G` is a polynomial used in
        :func:`newton_iteration` to generate additional terms in the
        y-series.
    """
    H = H.as_poly(x,y)
    R = []

    # when recurse is true, return if the leading order of H(0,y) is y
    if recurse:
        IH = H.subs(x,0).as_expr().leadterm(y)[1]
        if IH == 1:
            return [(H,x,y)]

    # for each newton polygon side branch out a new puiseux series
    data = newton_data(H,x,y,exceptional=(not recurse))
    R = []
    for q,m,l,phi in data:
        u,v = bezout(q,m)
        for psi,k in phi.factor_list()[1]:
            z = psi.gen
            psisimp = rootofsimp(psi)
            xi = psisimp.as_poly(z).root(0,radicals=False)
            Hprime = transform_newton_polynomial(H,x,y,q,m,l,xi)
            for (G,P,Q) in puiseux_rational(Hprime,x,y,recurse=True):
                foobar = (G, xi**v*P**q, P**m*(xi**u + Q))
                R.append(foobar)
    return R


def puiseux(f,x,y,alpha,beta=None,t=sympy.Symbol('t')):
    r"""Singular parts of the Puiseux series above :math:`x=\alpha`.

    Parameters
    ----------
    f : sympy.Expr
        A plane algebraic curve in `x` and `y`.
    alpha : complex
        The x-point over which to compute the Puiseux series of `f`.
    beta : complex
        (Optional) The y-point at which to compute the Puiseux series.
    t : sympy.Symbol
        (Optional) Variable used in the Puiseux series expansions.

    Returns
    -------
    list of PuiseuxTSeries

    """
    infinities = [sympy.oo, numpy.Inf, 'oo']
    singular = []

    # recenter the curve in x with the given alpha
    if alpha in infinities:
        alpha = sympy.oo
        d = degree(f,x)
        fa = expand(f.subs(x,1/x) * x**d)
    else:
        fa = expand(f.subs(x,x+alpha))

    # recenter the curve in y. if the curve is not monic, monicize and
    # perform the reverse transformation in the series construction step
    g,transform = almost_monicize(fa,x,y)
    gx0z = g.subs(x,0).subs(y,_Z).as_poly(_Z)
    all_roots = gx0z.all_roots(radicals=False,multiple=False)
    roots,multiplicities = zip(*all_roots)

    # for each y-center compute the singular parts of the puiseux series
    for beta in roots:
        H = g.subs(y,y+beta)
        singular_part_ab = puiseux_rational(H,x,y)

        # move back to (alpha, beta)
        for G,P,Q in singular_part_ab:
            if x in transform:
                Q /= transform.subs(x,P)
            else:
                Q += beta

            if alpha in infinities:
                P = 1/P
            else:
                P += alpha
            singular.append((G,P,Q))

    return singular


def almost_monicize(f,x,y):
    r"""Transform `f` to an "almost monic" polynomial.

    Perform a sequence of substitutions of the form

    .. math::

        f \mapsto x^d f(x,y/x)

    such that :math:`l(0) \neq 0` where :math:`l=l(x)` is the leading
    order coefficient of :math:`f`.

    Parameters
    ----------
    f,x,y : sympy.Expr
        An algebraic curve in `x` and `y`.

    Returns
    -------
    g, transform
        A new, almost monic polynomial `g` and a polynomial `transform`
        such that `y -> y/transform`.
    """
    f = f.expand()
    transform = sympy.S(1)
    monic = False
    while not monic:
        if LC(f,y).subs(x,0) == 0:
            fsubs = f.subs(y,y/x).expand()
            n,d = fsubs.together().as_numer_denom()
            f = n.expand()
            transform *= x
        else:
            monic = True
    return f,transform


# def puiseux(f, x, y, alpha, beta=None, t=sympy.Symbol('t'), parametric=False,
#             order=None, nterms=None, exact=True):
#     r"""Returns a list of Puiseux series lying above :math:`x=\alpha`.

#     Computes a list of :class:`PuiseuxTSeries` objects representing the
#     Puiseux series at the lift of :math:`x=\alpha`. Each of these series
#     can be further decomposed into :class:`PuiseuxXSeries` objects.

#     Parameters
#     ----------
#     f : sympy.Expr
#     x : sympy.Symbol
#     y : sympy.Symbol
#     alpha : complex
#         The x-point :math:`x=\alpha` at which to compute the set of
#         Puiseux series. `sympy.oo` is an allowed input
#     beta : complex
#         (Default: `None`) If provided, only returns the parametric
#         Puiseux series centered at :math:`(x,y) = (\alpha,\beta)`
#     var : sympy.Symbol
#         (Default: `sympy.Symbol('t')`) The variable in which the
#         parametric Puiseux series expansions are given.
#     parametric : bool
#         (Default: `False`) If `True`, returns a list of
#         `PuiseuxTSeries`. Otherwise, returns a list of `PuiseuxXSeries`.
#     order : int
#         (Default: `None`) If provided, returns Puiseux series expansions
#         up the the specified order.
#     nterms : int
#         (Default: `None`) If provided, returns Puiseux series expansions
#         with the specified number of terms.

#     Returns
#     -------
#     list
#         A list of :class:`PuiseuxTSeries` objects.

#     """
#     # scale and shift f accordingly: the algorithms for computing the
#     # Puiseux series terms are centered at x=0
#     if alpha in [sympy.oo, numpy.Inf, 'oo']:
#         alpha = sympy.oo
#         dx = sympy.degree(f,x)
#         fshift = f.subs(x, 1/x) * x**dx
#     else:
#         fshift = f.subs(x, x + alpha)
#     fshift = sympy.Poly(fshift,x,y).as_dict()

#     # compute the puiseux series data and construct the corresponding
#     # PuiseuxTSeries objects.
#     singular_data = singular(fshift,x,y,[])
#     series = [PuiseuxTSeries(f,x,y,alpha,datum,t=t,exact=exact)
#               for datum in singular_data]
#     for P in series:
#         P.extend(order=order, nterms=nterms)

#     # if y-value, beta, is given then return only the puiseux series
#     # such that Py(0) = beta
#     if beta is not None:
#         series = [Pi for Pi in series if Pi.eval_y(0) == beta]

#     # return x-series representations if requested
#     if not parametric:
#         series = [px for P in series for px in P.xseries()]
#     return series



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

    """
    @property
    def xdata(self):
        return (self.center, self.xcoefficient, self.ramification_index)
    @xdata.setter
    def xdata(self, value):
        self.center, self.xcoefficient, self.ramification_index = value

    @property
    def is_symbolic(self):
        return self._is_symbolic
    @property
    def is_numerical(self):
        return not self._is_symbolic

    @property
    def termsn(self):
        if self.is_numerical:
            return self.terms
        else:
            return [(numpy.int(n), numpy.complex(a)) for n,a in self.terms]
    @property
    def xdatan(self):
        if self.is_numerical:
            return self.xdata
        else:
            return (numpy.complex(self.center),
                    numpy.complex(self.xcoefficient),
                    numpy.int(self.ramification_index))

    def __init__(self, f, x, y, x0, singular_data, t=sympy.Symbol('t'),
                 order=None, exact=True):
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
        extension_terms, extension_polynomial = singular_data
        ramification_index, xcoefficient, terms = build_series(extension_terms)

        # store x-part attributes.
        sign = 1
        if x0 in [sympy.oo, numpy.Inf, 'oo']:
            x0 = 0
            sign = -1
        self.center = x0
        self.xcoefficient = xcoefficient
        self.ramification_index = sign*ramification_index

        # store y-part attributes and y-part extension data
        self.extension_terms = extension_terms
        self.extension_polynomial = extension_polynomial
        self.order = order
        self.terms = terms

        # coerce data to Numpy numerical types if requested on
        # construction
        self._is_symbolic = exact
        if self.is_numerical:
            self.coerce_to_numerical()

        # the curve, x-part, and terms output by puiseux make the
        # puiseux series unique. any mutability only adds terms
        self._hash = hash((self.f,
                           self.x0,
                           self.xcoefficient,
                           self.ramification_index,
                           tuple(self.terms)))

    def __repr__(self):
        """Print the x- and y-parts of the Puiseux series."""
        # print the x-part
        s = '%s(%s) = '%(self.x,self.t)
        s += '%s + '%(self.center) if self.center != 0 else ''
        s += '(%s)%s**%s\n'%(self.xcoefficient,self.t,self.ramification_index)

        # print the y-part
        s += '%s(%s) = '%(self.y,self.t)
        ss = '%s'%self.t
        for exp,coeff in self.terms:
            s += ' + (%s)'%coeff
            if exp != 0:
                s += ss + '**%s'%exp
        s += ' + O(%s**%s)'%(self.t, self.order)
        return s

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        r"""Check equality.

        A `PuiseuxTSeries` is uniquely identified by the curve it's
        defined on, its center, x-part terms, and the singular terms of
        the y-part.

        Parameters
        ----------
        other : PuiseuxTSeries

        Returns
        -------
        boolean

        """
        if is_instance(other, PuiseuxTSeries):
            if self._id == other._id:
                return True
        return False

    def coerce_to_numerical(self):
        r"""Coerces coefficients and data to numerical types.

        In numerical situations it is best to work with numerical types
        instead of symbolic ones for performance purposes. When
        `coerce_to_numerical` is executed all internal data structures
        are converted to Numpy data types.

        List of data coerced to numerical types:

        * x-part terms
        * y-part terms
        * y-part series extension data

        .. note::

            Symbolic coefficient data is lost once this is performed.

        Parameters
        ---
        None

        Returns
        ---
        None

        """
        # coerce x-part terms
        self.x0 = numpy.complex(self.x0)
        self.center = numpy.complex(self.center)
        self.xcoefficient = numpy.complex(self.xcoefficient)
        self.ramification_index = numpy.int(self.ramification_index)

        # coerce y-part terms and extension data
        self.terms = [(numpy.int(n), numpy.complex(a)) for n,a in self.terms]
        self.extension_terms = [
            (numpy.int(q),
            numpy.complex(mu),
            numpy.int(m),
            numpy.complex(beta),
            numpy.complex(eta))
            for (q,mu,m,beta,eta) in self.extension_terms
            ]
        self.extension_polynomial = dict(
            ((numpy.int(i),numpy.int(j)),numpy.complex(c))
            for ((i,j),c) in self.extension_polynomial.iteritems()
            )

        self._is_symbolic = False

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
        lamb = self.xcoefficient
        order = self.order

        # compute the e-th roots of lambda
        if self.is_symbolic:
            mu = sympy.root(1/lamb,e)
            conjugates = [mu*sympy.exp(2*sympy.pi*sympy.I*sympy.Rational(k,e))
                          for k in range(e)]
        else:
            e = numpy.double(e)
            mu = (1./lamb)**(1./e)
            conjugates = [mu*numpy.exp(2.0j*numpy.pi*k/e)
                          for k in range(int(e))]

        if not all_conjugates:
            conjugates = conjugates[0:1]

        # compute each conjugate x-series
        xseries = []
        for c in conjugates:
            terms = [(nh/e, alphah*c**nh) for nh,alphah in self.terms]
            p = PuiseuxXSeries(self.f,self.x,self.y,self.x0,terms,
                               order=order, ramification_index=int(e))
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
        return len(self.terms)

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
        bool
            Returns `True` if a new term was added. A term wouldn't be
            added if the Puiseux series expansion is finite.

        .. note::

            The calculation of the next :math:`\alpha` term involves a
            division that doesn't need to be there but avoiding this
            division requires some more state. The best way to hand this
            state is something that needs to be thought of.

        """
        # extract the tau data and get the current intermediate
        # polynomial F
        q,mu,m,beta,eta = self.extension_terms[-1]
        F = self.extension_polynomial
        x = self.x
        y = self.y

        # get the set of all exponents j where a_{0,j} x^j is a term in the
        # polynomial F. if this set is empty when we've found a finite
        # puiseux expansion.
        ms = [j for (j,i) in F.keys() if i==0 and j!=0]
        if not ms:
            return None

        # compute the next regular tau term and polynomial: m is equal to
        # the smallest degree x^m term and beta is the raito of the
        # coefficient of this term with the coefficient of y.
        m_next = min(ms)
        beta_next = -F[(m_next,0)]/F[(0,1)]
        tau = (1,1,m_next,beta_next,1)
        F = _new_polynomial(F,x,y,tau,m_next)

        # update the internal state, including the computation of the
        # actual y-series term and exponent
        self.extension_terms.append(tau)
        self.extension_polynomial = F

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
        r"""Extends the series in place.

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
            cmp = lambda terms: max(terms)[0]
            bound = order
        elif nterms:
            cmp = lambda terms: len(terms)
            bound = nterms

        # main loop: keep adding terms until the stopping condition is
        # satisfied. break if a finite Puiseux series expansion is
        # encountered
        while cmp(self.terms) < bound:
            if not self.add_term():
                break

    def extend_to_t(self, t, curve_tol=1e-8, rel_tol=1e-4):
        r"""Extend the series to accurately determine the y-values at `t`.

        Add terms to the t-series until the the regular place
        :math:`(x(t), y(t))` is within a particular tolerance of the
        curve that the Puiseux series is approximating.

        Parameters
        ----------
        t : complex
        eps : double
        curve_tol : double
            The tolerance for the corresponding point to lie on the curve.
        rel_tol : double
            A relative tolerance parameter used to ensure that the point
            :math:`(x(t_0),y(t_0))` is on the same branch as the center
            of the Puiseux series.

        Returns
        -------
        none
            The PuiseuxTSeries is modified in-place.
        """
        # note that we need to keep track of how much the y-value
        # changes with each iteration just in case it is intersecting
        # with a different branch of the curve
        num_iter = 0
        max_iter = 16
        while num_iter < max_iter:
            xt = sympy.N(self.eval_x(t))
            yt = sympy.N(self.eval_y(t))
            n,a = max(self.terms)
            a = a.n()

            curve_error = abs(sympy.N(self.f.subs({self.x:xt,self.y:yt})))
            rel_error = abs(sympy.N(a*t**n/yt))
            if (curve_error < curve_tol) and (rel_error < rel_tol):
                break
            else:
                self.add_term()
                num_iter += 1

    def extend_to_x(self, x, curve_tol=1e-8, rel_tol=1e-2):
        r"""Extend the series to accurately determine the y-values at `x`.

        Add terms to the t-series until the the regular place :math:`(x,
        y)` is within a particular tolerance of the curve that the
        Puiseux series is approximating.

        Parameters
        ----------
        x : complex
        curve_tol : double
            The tolerance for the corresponding point to lie on the curve.
        rel_tol : double
            A relative tolerance parameter used to ensure that the point
            :math:`(x(t_0),y(t_0))` is on the same branch as the center
            of the Puiseux series.

        Returns
        -------
        none
            The PuiseuxTSeries is modified in-place.
        """
        # simply convert to t and pass to extend. choose any conjugate
        # since the convergence rates between each conjugate is equal
        center, xcoefficient, ramification_index = self.xdata
        t = numpy.power((x-center)/xcoefficient, 1.0/ramification_index)
        self.extend_to_t(t, curve_tol=curve_tol, rel_tol=rel_tol)

    def eval_x(self, t):
        r"""Evaluate the x-part of the Puiseux series at `t`.

        Parameters
        ----------
        t : sympy.Expr or complex

        Returns
        -------
        complex

        """
        center, xcoefficient, ramification_index = self.xdata
        return center + xcoefficient*t**ramification_index

    def eval_dxdt(self, t):
        center, xcoefficient, ramification_index = self.xdata
        return xcoefficient*ramification_index*(1/t)**(1-ramification_index)

    def eval_y(self, t):
        r"""Evaluate of the y-part of the Puiseux series at `t`.

        Parameters
        ----------
        t : complex or complex

        Returns
        -------
        complex

        Notes
        -----

        This can be sped up using a Holder-like fast exponent evaluation
        trick.

        """
        return sum(alpha*t**n for n,alpha in self.terms)


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
    @property
    def order(self):
        return self._order
    @order.setter
    def order(self, value):
        # if order isn't specified then set order equal to the exponent
        # of the largest non-zero term plus 1/ramification_index
        if value:
            self._order = value
        else:
            order = max(exp for exp,coeff in self.terms)
            self._order = order + 1/self.ramification_index

        # truncate if new order is less than previous
        self.terms = tuple((exp,coeff) for exp,coeff in self.terms
                           if exp < self._order)
        self._hash = hash((self.f, self.x0, self._terms, self._order))

    @property
    def terms(self):
        return self._terms
    @terms.setter
    def terms(self, value):
        # filter out zero terms unless the series is the zero
        # series. (useful for accumulation.)
        terms = tuple((exp,coeff) for exp,coeff in value if coeff != 0)
        if not value:
            value = ((0,0),)
        self._terms = value

        # if order isn't set then assume all known terms are given. if
        # order ends up being zero then set to infinity
        if not self._order:
            order = max(exp for exp,coeff in self._terms)
            self._order = order if order else sympy.oo
        self._hash = hash((self.f, self.x0, self._terms, self._order))

    @property
    def is_symbolic(self):
        return self._is_symbolic
    @property
    def is_numerical(self):
        return not self._is_symbolic

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

        # intitialize data from x-part
        if x0 in [sympy.oo, numpy.Inf, 'oo']:
            x0 = 0
        self.center = x0
        self.ramification_index = None

        # intitalize terms from given object. coerce data to Numpy
        # numerical types if requested on construction.
        self._terms = None
        self._order = None
        terms = self.initialize_terms(obj, order=order)
        self.terms = tuple(sorted(terms, key=itemgetter(0)))

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
        ss = '%s'%self.x if self.center == 0 else '(%s)'%(self.x-self.center)
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
        if isinstance(obj,list):
            obj = tuple(obj)
        elif isinstance(obj,dict):
            obj = tuple(obj.items())
        if isinstance(obj, tuple):
            return obj
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
        terms = []
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
        return sum(alpha * (x-self.x0)**ne for ne,alpha in self.terms)

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


if __name__=='__main__':
    print '=================================='
    print '========== Test Puiseux =========='
    print '==================================\n'
    import sympy
    from sympy.abc import x,y,t
    import time

    # from csympy import *
    # x,y,t = var('x,y,t')
    # x = sympy.sympify(x)
    # y = sympy.sympify(y)
    # t = sympy.sympify(t)

    f1 = (x**2 - x + 1)*y**2 - 2*x**2*y + x**4
    f2 = -x**7 + 2*x**3*y + y**3
    f3 = (y**2-x**2)*(x-1)*(2*x-3) - 4*(x**2+y**2-2*x)**2
    f4 = y**2 + x**3 - x**2
    f5 = (x**2 + y**2)**3 + 3*x**2*y - y**3
    f6 = y**4 - y**2*x + x**2
    f7 = y**3 - (x**3 + y)**2 + 1
    f8 = (x**6)*y**3 + 2*x**3*y - 1
    f9 = 2*x**7*y + 2*x**7 + y**3 + 3*y**2 + 3*y
    f10 = (x**3)*y**4 + 4*x**2*y**2 + 2*x**3*y - 1
    f11 = y**2 - x*(x-1)*(x+1)

    f22 = y**3 - x**5
    f23 = (y - 1 - 2*x - x**2)*(y - 1 - 2*x - x**7)
    f27 = (y**2 - 2*x**3)*(y**2-2*x**2)*(y**3-2*x)

    g = x**4 - 2*x**2*y + y**2*(x**2 - x)

    f = f3

    f = expand(x**5*(y**2-1/x**5)*(y**3-x**7))

    for G,P,Q in puiseux(f,x,y,0):
        print '------------------------------------------------------------'
        P = rootofsimp(P)
        Q = rootofsimp(Q)
        G = rootofsimp(G)

        print '\n(x, y) =\n'
        sympy.pprint( (P,Q.subs(y,0).expand()) )
        print '\n(x(t), y(t)) =\n'

        g = newton_iteration(G,x,y,2)

        sympy.pprint(P)
        print
        sympy.pprint(Q)


    print '\n===== Done ====='
