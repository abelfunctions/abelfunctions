import sympy
from sympy.polys import (cancel, NotInvertible)


def rootofsimp(expr):
    """Reduce expressions containing ``RootOf`` types modulo their defining
    polynomials.

    A root `r` of a polynomial `p(x) = a_n x^n + ... + a_0` satisfies
    `p(r) = 0` by definition. Therefore, any power of `r` greater than
    `n` can be rewritten in terms of smaller powers. In general, any
    polynomial or rational expression of `r` can be reduced modulo its
    defining polynomial `p`.

    ``rootofsimp`` rewrites the expression as a numerator-denominator
    pair, computes the modular inverse of the denominator, and returns
    the product of this inverse with the numerator modulo the defining
    polynomial of the root. This is done for each ``RootOf`` appearing
    in the expression.

    Examples
    ========

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

    for root in expr.find(RootOf):
        # rewrite the defining polynomial in terms of the root, itself
        var = sympy.Dummy('root')
        modulus = root.poly.xreplace({root.poly.gen:var})
        numer,denom = cancel(expr.xreplace({root:var})).as_numer_denom()

        try:
            denom = denom.as_poly(var).invert(modulus)
            expr = numer.as_poly(var)*denom % modulus
        except NotInvertible:
            raise ZeroDivisionError('RootOf expression not invertible '
                                    'modulo %s'%modulus.as_expr())

        expr = expr.xreplace({var:root})
    return expr.as_expr()
