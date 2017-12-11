"""
Precision
============

Wrappers that allow globally setting a non-default precision for Sage's RealField and ComplexField

Authors
-------

* Timothy Olson (Dec 2017)
"""

from sage.all import RealField as RF
from sage.all import ComplexField as CF


PRECISION = None


def set_precision(prec):
    global PRECISION
    PRECISION = prec


def default_precision():
    global PRECISION
    PRECISION = None


def ComplexField(*pargs, **kwargs):
    """A wrapper around ComplexField that uses the current value of PRECISION."""
    field = CF(PRECISION) if PRECISION is not None else CF()
    
    if not pargs and not kwargs:
        return field
    else:
        return field(*pargs, **kwargs)

def RealField(*pargs, **kwargs):
    """A wrapper around RealField that uses the current value of PRECISION."""
    field = RF(PRECISION) if PRECISION is not None else RF()
    
    if not pargs and not kwargs:
        return field
    else:
        return field(*pargs, **kwargs)
