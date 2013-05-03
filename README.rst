Abelfunctions
=============

An Enthought Python Distribution / Sage library for computing with Abelian 
functions and solving integrable partial differential equations.

License: Abelfunctions is licensed under the New BSD License which covers
all files in the abelfunctions repository unless stated otherwise. See the
LICENSE file for details.

Documentation and Usage
-----------------------

See the Abelfunctions Wiki for full details.

To begin using Abelfunctions from this directory start python and run::

  >>> from abelfunctions import puiseux
  >>> from sympy.abc import x,y
  >>> from numpy import prod
  >>> f = y**3 - x**5
  >>> P = puiseux(f,x,y,0,5,parametric=False); P
  [(-x**(1/3)/2 + 3**(1/2)*I*x**(1/3)/2)**5,
   (-x**(1/3)/2 - 3**(1/2)*I*x**(1/3)/2)**5,
    x**(5/3)]
  >>> prod(y-pi for pi in P).simplify()
  -x**5 + y**3
    

Prerequisites
-------------

abelfunctions runs well with either the Enthought Python Distribution 
(http://enthought.com/products/epd.php) or Sage (http://www.sagemath.org).
Specifically, abelfunctions requires the following Python packages

  * numpy
  * scipy
  * sympy
  * networkx	      

Plotting / visual routines will require matplotlib, in addition.


Installation
------------

*Download the Code*. There are two ways to do this:

1) Download and extract a zipfile. First, go to the Abelfunctions
   homepage https://github.com/cswiercz/abelfunctions. Then, click on
   the button labeled "ZIP" near the top of the page.

2) If you have git (http://git-scm.com/) installed, run::

    $ git clone git://github.com/cswiercz/abelfunctions.git

*EPD Installation*. Enter the main directory, abelfunctions, and run::

  # python setup.py install

*Sage Installation*. Enter the main directory, abelfunctions, and run::

  $ sage -sh
  $ python setup.py install


Authors
-------

Chris Swierczewski <cswiercz@gmail.com>

