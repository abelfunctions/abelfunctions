#!/usr/bin/env python
"""Setup script for abelfunctions

To install Abelfunctions for your user account run:

    $ sage setup.py install --user

To build Abelfunctions in-place (used in running the test suite) run:

    $ sage setup.py build_ext --inplace

To install in a custom directory <dir> use:

    $ sage setup.py install --prefix=<dir>

Developers: to clean the directory of any extraneous files, such as compiled
Python .pyc and Cython .o/.so output and run:

    $ sage setup.py clean
"""
import os
import sys
import shutil
import unittest
from distutils.core import setup, Command
from distutils.extension import Extension
from Cython.Build import cythonize
from numpy.distutils.misc_util import get_numpy_include_dirs

# raise error if the user is not using Sage to compile
try:
    from sage.env import sage_include_directories
    SAGE_ROOT = os.environ['SAGE_ROOT']
except (ImportError, KeyError):
    raise EnvironmentError('abelfunctions must be built using Sage:\n\n'
                           '\t$ sage setup.py <args> <kwds>\n')

# list of Abelfunctions extension modules. most modules need to be compiled
# against the Sage and Numpy (included with Sage) libraries. The necessary
# include_dirs are provided after the module list.
#
ext_modules = [
    Extension('abelfunctions.complex_path',
              sources=[
                  os.path.join('abelfunctions',
                               'complex_path.pyx')]
          ),
    Extension('abelfunctions.riemann_surface_path',
              sources=[
                  os.path.join('abelfunctions',
                               'riemann_surface_path.pyx')]
          ),
    Extension('abelfunctions.puiseux_series_ring_element',
              sources=[
                  os.path.join('abelfunctions',
                               'puiseux_series_ring_element.pyx')]
          ),
    Extension('abelfunctions.riemann_theta.radius',
              sources=[os.path.join('abelfunctions','riemann_theta',
                                    'lll_reduce.c'),
                       os.path.join('abelfunctions','riemann_theta',
                                    'radius.pyx')]
          ),
    Extension('abelfunctions.riemann_theta.integer_points',
              sources=[os.path.join('abelfunctions','riemann_theta',
                                    'integer_points.pyx')]
          ),
    Extension('abelfunctions.riemann_theta.riemann_theta',
              sources=[os.path.join('abelfunctions','riemann_theta',
                                    'finite_sum.c'),
                       os.path.join('abelfunctions','riemann_theta',
                                    'riemann_theta.pyx')]
          ),
    ]

# parameters for all extension modules:
#
# * most modules depend on Sage and Numpy. Provide include directories.
# * disable warnings in gcc step
INCLUDES = sage_include_directories()
INCLUDES_NUMPY = get_numpy_include_dirs()
for mod in ext_modules:
    mod.include_dirs.extend(INCLUDES)
    mod.include_dirs.extend(INCLUDES_NUMPY)
    mod.extra_compile_args.append('-w')
    mod.extra_compile_args.append('-std=c99')

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
                    print('deleting %s...' % file)
                    os.remove(file)
                if file.endswith('~') and os.path.isfile(file):
                    print('deleting %s...' % file)
                    os.remove(file)

        os.chdir(dir_setup)

        # explicity remove files and directories from 'blacklist'
        blacklist = ['build', 'dist', 'doc/_build']
        for file in blacklist:
            if os.path.isfile(file):
                print('deleting %s...' % file)
                os.remove(file)
            elif os.path.isdir(file):
                print('deleting %s...' % file)
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
                    print('deleting %s...' % file)
                    os.remove(file_c)
                file_cpp = root + '.cpp'
                if os.path.isfile(file_cpp):
                    print('deleting %s...' % file)
                    os.remove(file_cpp)

        os.chdir(dir_setup)

        # delete cython .so modules
        ext_module_names = [ext.name for ext in ext_modules]
        for mod in ext_module_names:
            file = mod.replace('.', os.path.sep) + '.so'
            file = os.path.join(dir_setup, file)
            if os.path.isfile(file):
                print('deleting %s...' % file)
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
    license = 'MIT',
    packages = packages,
    install_requires=['six'],
    ext_modules = cythonize(ext_modules),
    platforms = ['all'],
    cmdclass = {'clean':clean}
)
