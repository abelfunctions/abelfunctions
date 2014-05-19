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

Installation Procedure
----------------------

1. **Download the Code**: There are two ways to do this:

  A) Go to the `abelfunctions repository
     <https://github.com/cswiercz/abelfunctions>`_, click on the button
     labeled "Download ZIP" located in the sidebar on the right, and
     extract the package.

  B) If you have `git <http://git-scm.com/>`_ installed, run ::

       $ git clone git://github.com/cswiercz/abelfunctions.git

     You can later update the code by running ::

       $ cd abelfunctions
       $ git pull

2. **Build**: Enter the main directory, ``abelfunctions``, and run ::

     $ python setup.py build_ext

3. **Install**: To make abelfunctions accessible by Python / iPython
   from any directory on your computer, run::

     $ python setup.py install

Installation for Developers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you plan on hacking at abelfunctions it may be easier build and
"install" abelfunctions in-place. To do so, run the following instead of
Steps 2 and 3 above ::

  $ python setup.py build_ext --inplace

and then run your favorite scientific Python distribution from the
``abelfunctions`` directory. If you make a change to any Cython
(``.pyx``, ``.pxd``) file you'll have to run the above command again and
re-import the ``abelfunctions`` package before being able to test your
changes.


Test Installation
-----------------

After installation, make sure you can import the abelfunctions package
in your favorite scientific Python distribution.

.. code-block:: python

  from abelfunctions import *
  from sympy.abc import x,y
  f = y**3 + 2*x**3*y - x**7
  X = RiemannSurface(f,x,y)
  print X

* .. code-block:: none

    Riemann surface defined by the algebraic curve -x**7 + 2*x**3*y + y**3
