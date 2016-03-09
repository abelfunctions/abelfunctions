Integral Bases of Algebraic Function Fields
===========================================


Computing Integral Bases
------------------------

    >>> import sympy
    >>> from sympy.abc import x,y
    >>> f = -x**7 + 2*x**3*y + y**3

    >>> from abelfunctions import integral_basis
    >>> b = integral_basis(f,x,y)
    >>> sympy.pprint(b)
    ⎡       2⎤
    ⎢   y  y ⎥
    ⎢1, ─, ──⎥
    ⎢   x   3⎥
    ⎣      x ⎦



References
----------

[1] Mark van Hoeij, *"Integral bases of algebraic function fields"*, 