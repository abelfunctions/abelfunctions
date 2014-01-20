Tutorial
========

(Put basic tutorial here.)

Riemann Surfaces
----------------

.. sourcecode:: ipython
   In [1]: from abelfunctions import *
   In [2]: from sympy.abc import x,y
   In [3]: f = y**3 + 2*x**3*y - x**7
   In [4]: C = RiemannSurface(f,x,y)
   In [5]: C
   Out[5]: Riemann surface defined by f(x,y) = y**3 + 2*x**3*y - x**7

Period Matrices
---------------

Abel Map
--------

Riemann Theta Functions
-----------------------

Plotting Examples
-----------------