Installation
============

.. _prerequisites:

Prerequisites
-------------

Abelfunctions runs well with either `Anaconda
<http://continuum.io/downloads.html>`_, the `Enthought Python Distribution
<http://enthought.com/products/epd.php>`_, or `Sage
<http://www.sagemath.org>`_.  Specifically, Abelfunctions requires the
following Python packages which are included in the aforementioned software:

* numpy
* scipy
* sympy
* networkx
* matplotlib
* cython

Additionaly, the following packages are required to build the documentation.

* sphinx
* numpydoc
* releases
* sphinx_rtd_theme


Installation Procedure
----------------------

1. **Download the Code**: There are two ways to do this:

  A) `Download Abelfunctions
     <https://github.com/cswiercz/abelfunctions/archive/master.zip>`_.

  B) If you have `git <http://git-scm.com/>`_ installed, run ::

       $ git clone https://github.com/cswiercz/abelfunctions.git

2. **Install**: Enter the top-level directory, `abelfunctions`, and run ::

     $ python setup.py install --user

   for a local installation. For a system-wide install, (or if you're
   installing the package into Sage) run ::

     $ python setup.py install
