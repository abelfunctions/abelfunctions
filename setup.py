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

__version__ = '1.0'

from distutils.core import setup, Command
from distutils.extension import Extension
from Cython.Distutils import build_ext
import unittest
import os
import os.path
import numpy

class clean(Command):
    """
    Cleans *.pyc and other trash files producing the same copy of the code
    as in the git repository.
    """

    description = "remove all build and trash files"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # helper function for deleting directory and contents
        def delete_dir(dir):
            # don't do anything if directory already doesn't exist
            if not os.path.isdir(dir):
                return

            # dive into directory contents. if file, delete.
            # if directory, run delete_dir.
            for f in os.listdir(dir):
                path = os.path.join(dir,f)
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                    elif os.path.isdir(path):
                        delete_dir(path)
                        os.removedirs(path)
                except Exception, e:
                    print e

        # deletes .pyc files that don't have corresponding .py files
        for root, dirs, files in os.walk('.'):
            pyc_files = filter(lambda f: f.endswith('.pyc'), files)
            tilde_files = filter(lambda f: f.endswith('~'), files)

            to_remove = pyc_files + tilde_files
            for f in to_remove:
                full_path = os.path.join(root,f)
                os.unlink(full_path)

        # delete build directory
        dir = 'build'
        delete_dir(dir)


class test_abelfunctions(Command):
    """
    Runs all tests under every abelfunctions/ directory.
    """
    description = "run all tests and doctests"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        loader = unittest.TestLoader()
        suite = loader.discover('abelfunctions')
        unittest.TextTestRunner(verbosity=2).run(suite)


packages = [
    'abelfunctions.riemanntheta',
    'abelfunctions.riemannsurface',
    'abelfunctions.utilities',
    ]

ext_modules = [
    Extension('abelfunctions.riemanntheta.riemanntheta_cy',
              sources = ['abelfunctions/riemanntheta/riemanntheta_cy.pyx',
                         'abelfunctions/riemanntheta/riemanntheta.c'],
              include_dirs = [numpy.get_include()]
              ),
    Extension('abelfunctions.riemannsurface.riemannsurface_path',
              sources = [
                  'abelfunctions/riemannsurface/riemannsurface_path.pyx',
                  ],
              include_dirs = [numpy.get_include()]
              ),
    Extension('abelfunctions.riemannsurface.smale',
              sources = [
                  'abelfunctions/riemannsurface/smale.pyx',
                  ],
              include_dirs = [numpy.get_include()]
              ),
    Extension('abelfunctions.riemanntheta.lattice_reduction',
              sources = ['abelfunctions/riemanntheta/lattice_reduction.pyx',
                         'abelfunctions/riemanntheta/lll_reduce.c'],
              include_dirs = [numpy.get_include()]
              ),
    ]


tests = [
    'abelfunctions.tests',
#    'abelfunctions.riemanntheta.tests',
    ]

classifiers = [
    'Programming Language :: Python',
    'Programming Language :: Cython',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics',
    'Intended Audience :: Science/Research',
    'Operating System :: Unix',
    'Operating System :: MaxOS'
    ]

long_description = '''
Abelfunctions is a Python library for computing with Abelian
functions. The primary goal of the application is to make computing with
Abelian functions as ubiquitous as computing with trigonometric
functions.  This framework is applied toward solving integrable systems
of partial differential equations. It is the research work of Chris
Swierczewski from the Department of Applied Mathematics at the
University of Washington.'''

setup(
    name = 'abelfunctions',
    version = __version__,
    description = 'Python library for computing with Abelian functions',
    long_description = long_description,
    author = 'Chris Swierczewski',
    author_email = 'cswiercz@gmail.com',
    url = 'https://github.com/cswiercz/abelfunctions',
    license = 'GPL v2+',
    packages = ['abelfunctions'] + packages + tests,
    ext_modules = ext_modules,
    cmdclass = {'test': test_abelfunctions,
                'clean': clean,
                'build_ext': build_ext
                },
    platforms = ['Linux', 'Unix', 'Mac OS-X'],
    classifiers = classifiers,
    )
