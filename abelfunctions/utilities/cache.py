"""
abelfunctions: cache.py

Code for cacheing functions.

Authors:

- Chris Swierczewski (November 2012)
"""

import collections
import functools

def cached_function(f):
    r"""Memoization decorator for functions taking multiple arguments.

    Parameters
    ----------
    f : function

    References
    ----------
    Taken from `http://code.activestate.com/recipes/578231-probably-the-fastest-memoization-decorator-in-the-/`.
    """
    class memodict(dict):
        def __getitem__(self, *key):
            return dict.__getitem__(self, key)

        def __missing__(self, key):
            ret = self[key] = f(*key)
            return ret

    return memodict().__getitem__


def cached_function_fast(f):
    r"""Memoization decorator for functions taking a single argument.

    Parameters
    ----------
    f : function

    References
    ----------
    Taken from `http://code.activestate.com/recipes/578231-probably-the-fastest-memoization-decorator-in-the-/`.
    """
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret
    return memodict().__getitem__


class cached_property(object):
    """Memoization decorator for class methods.

    The return value from a given method invocation will be cached on
    the instance whose method was invoked. All arguments passed to a
    method decorated with memoize must be hashable.

    If a memoized method is invoked directly on its class the result will not
    be cached.

    References
    ----------

    http://code.activestate.com/recipes/577452-a-memoize-decorator-for-instance-methods/
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        pobj = partial(self, obj)
        pobj.__doc__ = self.func.__doc__
        pobj.__name__ = self.func.__name__
        pobj.__module__ = self.func.__module__
        return pobj

    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}

        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res

