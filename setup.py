#!/usr/bin/env python
"""
Distutils based setup script for abelfunctions

This uses Distutils (http://python.org/sigs/distutils-sig/) the standard
python mechanism for installing packages. For the easiest installation
just type the command (you'll probably need root privileges for that):

    python setup.py install

This will install the library in the default location. To install in a
custom directory <dir>, use:

    python setup.py install --prefix=<dir>

To install for your user account (recommended) use:

    python setup.py install --user

"""

__version__ = '1.0'

import numpy

import glob
import os
import shutil
import unittest

from distutils.core import setup, Command
from distutils.extension import Extension
from Cython.Distutils import build_ext

class clean(Command):
    """Cleans files so you should get the same copy as in git."""
    description = 'remove build files'
    user_options = [('all', 'a', 'the same')]

    def initialize_options(self):
        self.all = None

    def finalize_options(self):
        pass

    def run(self):
        # delete all files ending with certain extensions
        # currently: '.pyc', '~'
        dir_setup = os.path.dirname(os.path.realpath(__file__))
        curr_dir = os.getcwd()
        for root, dirs, files in os.walk(dir_setup):
            for file in files:
                file = os.path.join(root, file)
                if file.endswith('.pyc') and os.path.isfile(file):
                    os.remove(file)
                if file.endswith('~') and os.path.isfile(file):
                    os.remove(file)

        os.chdir(dir_setup)

        # explicity remove files and directories from 'blacklist'
        blacklist = ['build', 'dist', 'doc/_build']
        for file in blacklist:
            if os.path.isfile(file):
                os.remove(file)
            elif os.path.isdir(file):
                shutil.rmtree(file)

        os.chdir(dir_setup)

        # delete temporary cython .c files. be careful to only delete the .c
        # files corresponding to .pyx files. (keep other .c files)
        ext_sources = [f for ext in ext_modules for f in ext.sources]
        for file in ext_sources:
            file = os.path.join(dir_setup, file)
            if file.endswith('.pyx') and os.path.isfile(file):
                (root, ext) = os.path.splitext(file)
                file_c = root + '.c'
                if os.path.isfile(file_c):
                    os.remove(file_c)

        os.chdir(dir_setup)

        # delete cython .so modules
        ext_module_names = [ext.name for ext in ext_modules]
        for mod in ext_module_names:
            file = mod.replace('.', os.path.sep) + '.so'
            file = os.path.join(dir_setup, file)
            if os.path.isfile(file):
                os.remove(file)

        os.chdir(curr_dir)



class test_abelfunctions(Command):
    """Runs all tests under every abelfunctions/ directory."""
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
    'abelfunctions.utilities',
    ]

ext_modules = [
    Extension('abelfunctions.abelmap',
              sources = ['abelfunctions/abelmap.pyx'],
              include_dirs = [numpy.get_include()],
              extra_compile_args = ['-Wno-unused-function']),
    Extension('abelfunctions.riemann_surface',
              sources = ['abelfunctions/riemann_surface.pyx'],
              extra_compile_args = ['-Wno-unused-function']),
    Extension('abelfunctions.riemann_surface_path',
              sources = ['abelfunctions/riemann_surface_path.pyx'],
              include_dirs = [numpy.get_include()],
              extra_compile_args = ['-Wno-unused-function']),
    Extension('abelfunctions.analytic_continuation',
              sources = ['abelfunctions/analytic_continuation.pyx'],
              include_dirs = [numpy.get_include()],
              extra_compile_args = ['-Wno-unused-function']),
    Extension('abelfunctions.analytic_continuation_smale',
              sources = ['abelfunctions/analytic_continuation_smale.pyx'],
              include_dirs = [numpy.get_include()],
              extra_compile_args = ['-Wno-unused-function']),
    Extension('abelfunctions.polynomials',
              sources = ['abelfunctions/polynomials.pyx'],
              include_dirs = [numpy.get_include()],
              extra_compile_args = ['-Wno-unused-function']),
    Extension('abelfunctions.differentials',
              sources = ['abelfunctions/differentials.pyx'],
              include_dirs = [numpy.get_include()],
              extra_compile_args = ['-Wno-unused-function']),
    Extension('abelfunctions.riemanntheta.radius',
              sources = ['abelfunctions/riemanntheta/lll_reduce.c',
                         'abelfunctions/riemanntheta/radius.pyx'],
              include_dirs = [numpy.get_include()],
              extra_compile_args = ['-Wno-unused-function']),
    Extension('abelfunctions.riemanntheta.integer_points',
              sources = ['abelfunctions/riemanntheta/integer_points.pyx'],
              include_dirs = [numpy.get_include()],
              extra_compile_args = ['-Wno-unused-function']),
    Extension('abelfunctions.riemanntheta.riemann_theta',
              sources = ['abelfunctions/riemanntheta/finite_sum.c',
                         'abelfunctions/riemanntheta/riemann_theta.pyx'],
              include_dirs = [numpy.get_include()],
              extra_compile_args = ['-Wno-unused-function']),
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
Abelfunctions is a library for computing with Abelian functions, Riemann
surfaces, and algebraic curves. The primary goal of the application is to make
computing with Abelian functions as ubiquitous as computing with trigonometric
functions.  This framework is applied toward solving integrable systems of
partial differential equations. It is the research work of Chris Swierczewski
from the Department of Applied Mathematics at the University of Washington.
'''

setup(
    name = 'abelfunctions',
    version = __version__,
    description = 'A library for computing with Abelian functions, Riemann '
                  'surfaces, and algebraic curves.',
    long_description = long_description,
    author = 'Chris Swierczewski',
    author_email = 'cswiercz@gmail.com',
    url = 'https://github.com/cswiercz/abelfunctions',
    license = 'GPL v2+',
    packages = ['abelfunctions'] + packages + tests,
    ext_modules = ext_modules,
    cmdclass = {'test': test_abelfunctions,
                'clean': clean,
                'build_ext': build_ext,
                },
    platforms = ['Linux', 'Unix', 'Mac OS-X'],
    classifiers = classifiers,
    )
