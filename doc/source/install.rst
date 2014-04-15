Installation
============

Prerequisites
-------------

abelfunctions runs well with either `Anaconda
<http://continuum.io/downloads.html>`_, the `Enthought Python
Distribution <http://enthought.com/products/epd.php>`_, or `Sage
<http://www.sagemath.org>`_.  Specifically, abelfunctions requires the
following Python packages:

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

**Download the Code**: There are two ways to do this:

A) Download and extract a zipfile. First, go to the `abelfunctions
   homepage <https://github.com/cswiercz/abelfunctions>`_. Then, click
   on the button labeled "Download ZIP" located in the sidebar on the
   right.

B) If you have `git <http://git-scm.com/>`_ installed, run::

    $ git clone git://github.com/cswiercz/abelfunctions.git

**Installation**: Enter the main directory, `abelfunctions`, and run::

  $ python setup.py install


Test Installation
-----------------

After installation, make sure you can import the abelfunctions package
in your favorite Python distribution:

.. code-block:: python

      $ ipython
      In [1]: from abelfunctions import *
      In [2]: from sympy.abc import x,y
      In [3]: f = y**3 + 2*x**3*y - x**7
      In [4]: X = RiemannSurface(f,x,y)
      In [5]: print X

*
    .. code-block:: none

        Riemann surface defined by the algebraic curve -x**7 + 2*x**3*y + y**3
