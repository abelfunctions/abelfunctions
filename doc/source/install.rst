Installation
============

Prerequisites
-------------

abelfunctions runs well with either `Anaconda
<http://continuum.io/downloads.html>`_, the `Enthought Python
Distribution <http://enthought.com/products/epd.php>`_, or `Sage
<http://www.sagemath.org>`_.  Specifically, abelfunctions requires the
following Python packages

* gcc or clang
* numpy
* scipy
* sympy
* networkx
* matplotlib
* cython

Optionally, the NVIDIA CUDA compiler is needed to compile the high-performance
CUDA code used in computing the Riemann theta function.

Installation Options
--------------------

*Download the Code*. There are two ways to do this:

1) Download and extract a zipfile. First, go to the `abelfunctions
   homepage <https://github.com/cswiercz/abelfunctions>`_. Then, click on
   the button labeled "ZIP" near the top of the page.

2) If you have `git <http://git-scm.com/>`_ installed, run::

    $ git clone git://github.com/cswiercz/abelfunctions.git

*Anaconda or EPD Installation*. Enter the main directory, `abelfunctions`, and run::

  $ python setup.py install

*Sage Installation*. Enter the main directory, abelfunctions, and run::

  $ sage -sh
  $ python setup.py install


Test Installation
-----------------

After installation, make sure you can import the `abelfunctions` package::

      $ python
      >>> from abelfunctions import *
      >>> from sympy.abc import x,y
      >>> C = RiemannSurface(y**2 - x**3 + 1, x, y)
      >>> print(C)