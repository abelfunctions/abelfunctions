Puiseux Series
==============

Puiseux series are simply formal series in fractional powers. They are
used to describe local expansions of places on an algebraic curve
including at branch points and singular points.

The algorithm imlpemented in `abelfunctions` is based on Duval [1].

Computing Puiseux Series
------------------------

To compute Puiseux series using abelfunctions, begin by defining a
polynomial in two variables.

    >>> import sympy
    >>> from sympy.abc import x,y,T
    >>> f = y**8 + x*y**5 + x**4 - x**6

The curve f above is irreducible over Q[x,y] and has eight sheets. For
ease of reading we compute a parameterized form of the Puiseux series:

    >>> from abelfunctions import puiseux
    >>> p = puiseux(f,x,y,a=0,nterms=3,parametric=T)
    >>> for pi in p: sympy.pprint(pi)
    ⎛         11    7     ⎞
    ⎜ 5    6⋅T     T     3⎟
    ⎜T , - ───── - ── - T ⎟
    ⎝        25    5      ⎠
    ⎛          9    5    ⎞
    ⎜  3    2⋅T    T     ⎟
    ⎜-T , - ──── - ── + T⎟
    ⎝        3     3     ⎠



References
----------

[1] Dominique Duval, *"Rational Puiseux expansions"*, *Composito Mathematica* v. 70, no. 2 (1989), p. 119-154.