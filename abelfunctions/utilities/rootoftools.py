r"""RootOf Tools :mod:`abelfunctions.utilities.rootoftools`
=======================================================

Various tools (on top of what Sympy provides) for dealing with ``RootOf``
objects.

Sympy's ``RootOf`` construct still lacks some features. Until the functionality
in this module is incorporated into Sympy itself it will be defined
here. (Since it's very necessary for the computations in this software.) The
primary motivation behind writing this functionality come from:

* Sympy not preserving the ``radicals=False`` specification in ``RootOf``
  construction.

* The lack of a simplification function: expresssions involving powers of
  ``RootOf`` can be rewritten such that all powers larger than the degree of
  the defining polynomial can be rewritten in terms of lower order terms. This
  is useful for managing expression swell.

Functions
---------

.. autosummary::

    all_roots
    rootofsimp

References
----------

.. [CLO] D. Cox, J. Little, D. O'Shea, "Using algebraic geometry",
   Springer-Verlag New York, New York, 2005.

Examples
--------

Contents
--------

"""

import sympy

from sympy import sympify, Dummy, resultant
from sympy.polys import cancel, NotInvertible

from .cache import cached_function


@cached_function
def all_roots(f, gen, multiple=True, radicals=False, tol=1e-15):
    r"""Return a list of real and complex roots with multiplicities.

    Modified from ``sympy.all_roots()`` to accept polynomials with a ``RootOf``
    type appearing in the coefficients.

    .. note::

        At the moment, this function only supports polynomials where the
        ``RootOf`` s appearing in the coefficients is the same root. That is,
        this function only supports polynomials in :math:`\mathbb{Q}[x,\xi]`
        where :math:`\xi` is **a root** of some polynomial.

        A future version should use multipolynomial resultants to handle
        multiple roots. (See Chapter 3.2 of [CLO]_.)

    Parameters
    ----------
    f : polynomial
    gen : Sympy symbol
        The dependent variable of the polynomial.

    Returns
    -------
    roots : list
        The roots of f as ``RootOf`` objects.

    Examples
    --------
    >>> from sympy import RootOf
    >>> from sympy.abc import x,y
    >>> from abelfunctions.utilities import all_roots
    >>> a = RootOf(y**2 + 1, 0, radicals=False)
    >>> f = x**2 + a*x - 1
    >>> all_roots(f,x)
    [RootOf(_x**4 - _x**2 + 1, 1), RootOf(_x**4 - _x**2 + 1, 3)]
    >>> map(lambda z: f.evalf(subs={x:z}), all_roots(f,x))
    [0.e-126 + 0.e-128*I, 0.e-126 - 0.e-128*I]


    """
    from sympy.polys.rootoftools import RootOf
    f = sympify(f).as_poly(gen)
    rootofs = f.find(RootOf)
    if len(rootofs) == 0:
        return f.all_roots(multiple=multiple, radicals=radicals)
    if len(rootofs) > 1:
        raise NotImplementedError('sorted roots not supported over multiple '
                                  'extensions')

    # comptue the resultant of the polynomial f and the defining polynomial of
    # the RootOf. rewrite in terms of a dummy variable.
    #
    # ptilde = the defining polynomial of alpha
    # ftilde = the input polynomial with the rootof replaced by a dummy varaible
    _gen = Dummy(str(gen))
    alphavar = Dummy()
    alpha = rootofs.pop()
    ptilde = alpha.poly.as_expr().xreplace({alpha.poly.gen:alphavar})
    ftilde = f.as_expr().xreplace({alpha:alphavar})
    res = resultant(ftilde, ptilde, alphavar).xreplace({gen:_gen}).as_poly(_gen)

    # compute the roots of the resultant and keep the ones which are zeros of
    # the input polynomial f (up to tolerance)
    res_roots = res.all_roots(multiple=True, radicals=False)
    is_root = lambda root: abs(f.evalf(subs={gen:root})) < tol
    roots = filter(is_root, res_roots)
    return roots


def rootofsimp(expr):
    r"""Symplifies expressions containing ``RootOf`` types.

    Reduces expressions containing ``RootOf`` types modulo their defining
    polynomials. In particualr, a root :math:`r` of a polynomial :math:`p(x) =
    a_n x^n + ... + a_0` satisfies :math:`p(r) = 0` by definition. Therefore,
    any power of :math:`r` greater than :math:`n` can be rewritten in terms of
    smaller powers. In general, any polynomial or rational expression of
    :math:`r` can be reduced modulo its defining polynomial :math:`p`.

    :func:`rootofsimp` rewrites the expression as a numerator-denominator pair,
    computes the modular inverse of the denominator, and returns the product of
    this inverse with the numerator modulo the defining polynomial of the
    root. This is done for each ``RootOf`` appearing in the expression.

    Parameters
    ----------
    expr : Sympy expression

    Returns
    -------
    expr : Sympy expression
        The input expression reduced modulo the defining polynomials of the
        ``RootOf`` s appearing within.

    Examples
    --------
    >>> from sympy import rootofsimp, RootOf
    >>> from sympy.abc import x,y
    >>> r = RootOf(x**4 - x - 1, 0)
    >>> rootofsimp(r**4)
    RootOf(x**4 - x - 1, 0) + 1
    >>> rootofsimp(r**8)
    2*RootOf(x**4 - x - 1, 0) + RootOf(x**4 - x - 1, 0)**2 + 1
    >>> rootofsimp(1/r)
    -1 + RootOf(x**4 - x - 1, 0)**3

    """
    from sympy.polys.rootoftools import RootOf

    expr = sympy.sympify(expr).as_expr()

    # temporarily replace every RootOf with a dummy variable. this is needed to
    # preserve options such as radical=True when simplifying
    rootofs = expr.find(RootOf)
    dummies = [sympy.Dummy() for _ in rootofs]
    transform = dict(zip(rootofs,dummies))
    expr = expr.xreplace(transform)

    # simplify in each root
    for root,dummy in zip(rootofs,dummies):
        try:
            modulus = root.poly.xreplace({root.poly.gen:dummy})
            numer,denom = cancel(expr).as_numer_denom()
            denom = denom.as_poly(dummy).invert(modulus)
            expr = numer.as_poly(dummy)*denom % modulus
        except NotInvertible:
            raise ZeroDivisionError('RootOf expression not invertible '
                                    'modulo %s'%modulus.as_expr())
        except TypeError:
            # most likely this exception occurred because expr is not a
            # polynomial expression in the rootof.
            pass

    # put the RootOfs back in place of the dummy substitutions
    transform = dict(zip(dummies,rootofs))
    expr = expr.xreplace(transform)
    return expr.as_expr()
