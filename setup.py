#!/usr/bin/env python
"""Distutils based setup script for abelfunctions

This uses Distutils (http://python.org/sigs/distutils-sig/) the standard python
mechanism for installing packages. For the easiest installation just type the
command (you'll probably need root privileges for that):

    $ python setup.py install

This will install the library in the default location. To install in a
custom directory <dir>, use:

    $ python setup.py install --prefix=<dir>

To install for your user account (recommended) use:

    $ python setup.py install --user

Finally, if you wish to build

"""
import os
import sys
import shutil
import unittest
from distutils.core import setup, Command
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy.distutils.misc_util import get_numpy_include_dirs

# include every conceivable directory that may contains sage headers
try:
    SAGE_ROOT = os.environ['SAGE_ROOT']
    SAGE_LOCAL = os.environ['SAGE_LOCAL']
except KeyError:
    raise EnvironmentError('abelfunctions must be built using Sage:\n\n'
                           '\t$ sage setup.py <args> <kwds>\n')

INCLUDES = [os.path.join(SAGE_ROOT),
            os.path.join(SAGE_ROOT,'src'),
            os.path.join(SAGE_ROOT,'src','sage'),
            os.path.join(SAGE_ROOT,'src','sage','ext'),
            os.path.join(SAGE_ROOT,'src','sage','ext','interrupt'),
            os.path.join(SAGE_LOCAL,'include'),
            os.path.join(SAGE_LOCAL,'include','python')]
INCLUDES_NUMPY = get_numpy_include_dirs()

ext_modules = [
    Extension('abelfunctions.complex_path',
              sources=[
                  os.path.join('abelfunctions',
                               'complex_path.pyx')],
              extra_compile_args = ['-std=c99'],
          ),
    Extension('abelfunctions.riemann_surface_path',
              sources=[
                  os.path.join('abelfunctions',
                               'riemann_surface_path.pyx')],
              extra_compile_args = ['-std=c99'],
          ),
    Extension('abelfunctions.puiseux_series_ring_element',
              sources=[
                  os.path.join('abelfunctions',
                               'puiseux_series_ring_element.pyx')],
              extra_compile_args = ['-std=c99'],
          ),
    Extension('abelfunctions.riemann_theta.radius',
              sources=[os.path.join('abelfunctions','riemann_theta',
                                    'lll_reduce.c'),
                       os.path.join('abelfunctions','riemann_theta',
                                    'radius.pyx')],
              extra_compile_args = ['-std=c99'],
          ),
    Extension('abelfunctions.riemann_theta.integer_points',
              sources=[os.path.join('abelfunctions','riemann_theta',
                                    'integer_points.pyx')],
              extra_compile_args = ['-std=c99'],
          ),
    Extension('abelfunctions.riemann_theta.riemann_theta',
              sources=[os.path.join('abelfunctions','riemann_theta',
                                    'finite_sum.c'),
                       os.path.join('abelfunctions','riemann_theta',
                                    'riemann_theta.pyx')],
              extra_compile_args = ['-std=c99'],
          ),
    ]

# parameters for all extension modules:
#
# * use all include directories in INCLUDES
# * disable warnings in gcc step
for mod in ext_modules:
    mod.include_dirs.extend(INCLUDES)
    mod.include_dirs.extend(INCLUDES_NUMPY)
    mod.extra_compile_args.append('-w')

# package and sub-package list
packages = [
    'abelfunctions',
    'abelfunctions.riemann_theta',
    'abelfunctions.utilities'
]

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
                    print 'deleting %s...'%file
                    os.remove(file)
                if file.endswith('~') and os.path.isfile(file):
                    print 'deleting %s...'%file
                    os.remove(file)

        os.chdir(dir_setup)

        # explicity remove files and directories from 'blacklist'
        blacklist = ['build', 'dist', 'doc/_build']
        for file in blacklist:
            if os.path.isfile(file):
                print 'deleting %s...'%file
                os.remove(file)
            elif os.path.isdir(file):
                print 'deleting %s...'%file
                shutil.rmtree(file)

        os.chdir(dir_setup)

        # delete temporary cython .c and .cpp files. be careful to only delete
        # the .c and .cpp files corresponding to .pyx files.
        ext_sources = [f for ext in ext_modules for f in ext.sources]
        for file in ext_sources:
            file = os.path.join(dir_setup, file)
            if file.endswith('.pyx') and os.path.isfile(file):
                (root, ext) = os.path.splitext(file)
                file_c = root + '.c'
                if os.path.isfile(file_c):
                    print 'deleting %s...'%file
                    os.remove(file_c)
                file_cpp = root + '.cpp'
                if os.path.isfile(file_cpp):
                    print 'deleting %s...'%file
                    os.remove(file_cpp)

        os.chdir(dir_setup)

        # delete cython .so modules
        ext_module_names = [ext.name for ext in ext_modules]
        for mod in ext_module_names:
            file = mod.replace('.', os.path.sep) + '.so'
            file = os.path.join(dir_setup, file)
            if os.path.isfile(file):
                print 'deleting %s...'%file
                os.remove(file)

        os.chdir(curr_dir)

# configure setup
exec(open('abelfunctions/version.py').read())
setup(
    name = 'abelfunctions',
    version = __version__,
    description = 'A library for computing with Abelian functions, Riemann '
                  'surfaces, and algebraic curves.',
    author = 'Chris Swierczewski',
    author_email = 'cswiercz@gmail.com',
    url = 'https://github.com/cswiercz/abelfunctions',
    license = 'GPL v2+',
    packages = packages,
    ext_modules = ext_modules,
    platforms = ['all'],
    cmdclass = {'clean':clean, 'build_ext':build_ext},
)
