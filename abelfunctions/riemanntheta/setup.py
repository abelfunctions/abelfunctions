#Grady Williams (gradyrw@uw.edu) October - 2012

"""
Setup.py file for RiemannCY, the cython wrapper function
for riemanntheta.c

"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension('RIEMANN',
	sources = ["Cython-Riemann.pyx", "riemanntheta.c"],
	)
	]

setup (
	ext_modules = ext_modules,
	cmdclass = {'build_ext': build_ext},
	)
