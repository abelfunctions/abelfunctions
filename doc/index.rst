.. abelfunctions documentation master file, created by
   sphinx-quickstart on Mon Oct 14 16:33:40 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

abelfunctions Documentation
===========================

abelfunctions is a Python package for computing with Abelian functions
and complex algebraic curves. It is the Ph.D. thesis work of Chris
Swierczewski. (www.cswiercz.info)

Source code is available at http://github.com/cswiercz/abelfunctions.

Quickstart
----------

Here is a quick introduction to computing period matrices of a complex
algebraic curve using abelfunctions. Additional documentation can be
found in the section below.

This code computes a period matrix of the algebraic curve :math:`f(x,y)
= -x^7 + 2x^3y + y^3`.

    >>> from abelfunctions import *
    >>> from sympy.abc import x,y
    >>> f = -x**7 + 2*x**3*y + y**3
    >>> X = RiemannSurface(f,x,y)
    >>> A,B = X.period_matrix()
    >>> print A
    [[ -1.38175582e-12-1.20192474j   1.84957199e+00+0.60096237j]
     [  9.22917298e-12+1.97146395j   7.16176201e-01-0.98573197j]]
    >>> print B
    [[-0.70647363+2.17430227j -1.84957199+2.54571744j]
     [-1.87497364-1.36224808j -0.71617620+0.23269975j]]

Documentation
-------------

.. toctree::
   :maxdepth: 2



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

