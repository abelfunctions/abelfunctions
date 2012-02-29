#!/usr/bin/env python
"""
Distutils based setup script for abelfunctions

This uses Distutils (http://python.org/sigs/distutils-sig/) the standard
python mechanism for installing packages. For the easiest installation
just type the command (you'll probably need root privileges for that):

    python setup.py install

This will install the library in the default location. To install in a
custom directory <dir>, use

    python setup.py --prefix=<dir>
"""

from distutils.core import setup
import abelfunctions

packages = [
    'abelfunctions',
    'abelfunctions.puiseux',
    'abelfunctions.monodromy',
    'abelfunctions.homology',
    'abelfunctions.riemannsurface',
    ]

classifiers = [
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics',
    ]

setup(
    name = 'abelfunctions',
    version = abelfunctions.__version__,
    description = 'Python library for computing with Abelian functions',
    author = 'Chris Swierczewski',
    author_email = 'cswiercz@gmail.com',
    license = 'GPL v2+',
    packages = packages,
    ext_modules = [],
    classifiers = classifiers
    )
