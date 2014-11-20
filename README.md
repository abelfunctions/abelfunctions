Abelfunctions
=============
[![Gitter](https://badges.gitter.im/Join Chat.svg)](https://gitter.im/cswiercz/abelfunctions?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

A library for computing with Abelian functions, Riemann surface, and
algebraic curves.  See the
[Documentation](http://abelfunctions.cswiercz.info) for more
information.

*Note: abelfunctions is still in early stages of development. Any issues
should be reported to the
[Issues Page](https://github.com/cswiercz/abelfunctions/issues) of the
project.*

Prerequisites
-------------

abelfunctions runs well with either
[Anaconda](https://store.continuum.io/cshop/anaconda/), the [Enthought
Python Distribution](http://enthought.com/products/epd.php) or
[Sage](http://www.sagemath.org).  Specifically, abelfunctions requires
the following Python packages

* gcc (or equivalent)
* numpy
* scipy
* sympy
* networkx
* matplotlib
* Cython

Optionally, the NVIDIA CUDA compiler is needed to compile the
high-performance CUDA code used in RiemannTheta.


Installation
------------

**Download the Code**. There are two ways to do this.

1. Download and extract a zipfile. First, go to the Abelfunctions
   homepage https://github.com/cswiercz/abelfunctions. Then, click on
   the button labeled "ZIP" near the top of the page.

2. If you have git (http://git-scm.com/) installed, run:

        $ git clone https://github.com/cswiercz/abelfunctions.git

   and it will download as `abelfunctions` in the current directory.

**Installation**. Enter the main directory, abelfunctions, and run:

    $ python setup.py build_ext --inplace

for a local (in-place) installation. For a system-wide install, run:

    $ python setup.py build_ext
    $ python setup.py install

See the documentation for more information.

Authors
-------

* Chris Swierczewski <cswiercz@gmail.com>
* Grady Williams
* James Collins
