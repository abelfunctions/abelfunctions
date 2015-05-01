Abelfunctions
=============

A library for computing with Abelian functions, Riemann surfaces, and algebraic
curves. Please see the [Documentation](http://abelfunctions.cswiercz.info) for
more information.

> **Warning:** abelfunctions is still in active development. Any issues should
  be reported to the [Issues
  Page](https://github.com/cswiercz/abelfunctions/issues) of the project or you
  can contact Chris Swierczewski directly at <cswiercz@gmail.com>.

**Chat Room**

[![Gitter](https://badges.gitter.im/Join Chat.svg)](https://gitter.im/cswiercz/abelfunctions?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) Chat with the developers, ask questions, provide feedback.

**Build Status**

[![Build Status](https://travis-ci.org/cswiercz/abelfunctions.svg?branch=master)](https://travis-ci.org/cswiercz/abelfunctions) Master Branch

[![Build Status](https://travis-ci.org/cswiercz/abelfunctions.svg?branch=dev)](https://travis-ci.org/cswiercz/abelfunctions) Dev Branch


Prerequisites
-------------

abelfunctions runs well with either
[Anaconda](https://store.continuum.io/cshop/anaconda/), the [Enthought Python
Distribution](http://enthought.com/products/epd.php) or with
[Sage](http://www.sagemath.org). See the
[documentation](http://abelfunctions.cswiercz.info) for an explicit list of
prerequisites.

Optionally, the NVIDIA CUDA compiler is needed to compile the high-performance
CUDA code used in computing the Riemann theta function.


Installation
------------

**Download the Code**. There are two ways to do this.

1. Download and extract a zipfile. First, go to the Abelfunctions homepage
   https://github.com/cswiercz/abelfunctions. Then, click on the button labeled
   "ZIP" near the top of the page.

2. If you have git (http://git-scm.com/) installed, run:

        $ git clone https://github.com/cswiercz/abelfunctions.git

   and it will download as `abelfunctions` in the current directory.

**Installation**. Enter the main directory, abelfunctions, and run:

    $ python setup.py install --user

for a local installation. For a system-wide install, (or if you're installing
the package into Sage) run:

    $ python setup.py install

Authors
-------

* (Primary) Chris Swierczewski (<cswiercz@gmail.com>)
* Grady Williams - CUDA / GPU Riemann theta
* James Collins

Citing this Software
--------------------

How to cite:

> C. Swierczewski et. al., *abelfunctions: A library for computing with Abelian
  functions, Riemann surfaces, and algebraic curves*,
  `http://abelfunctions.cswiercz.info`, 2015.

BibTeX:

    @misc{abelfunctions,
      author = {C. Swierczewski and others},
      title = {abelfunctions: A library for computing with Abelian functions, Riemann surfaces, and algebraic curves},
      note= {\tt http://abelfunctions.cswiercz.info},
      year = 2015,
    }
