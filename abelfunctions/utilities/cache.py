r"""Cache :mod:`abelfunctions.utilities.cache`
==========================================

Module defining cached function decorators.

The decorator :func:`cached_function` works with instance methods as well. It
relies on the ``decorator`` module for forwarding function signatures and
documentation. (Results in poorer caching performance than some other decorator
designs but doesn't break the reference documenation.)

Functions
---------

.. autosummary::

    cached_function
    cached_function_fast
    cached_method

Examples
--------

Contents
--------

"""

import decorator

def cached_function(obj):
    r"""Decorator for argument and keyword caching.

    This memoizing decorator caches over arguments as well as
    keywords. Including keywords does come at a large performance cost but is
    completely general.

    The use of ``decorator.decorator`` ensures that function signatures,
    documentation, etc. pass along to the decorated function.

    Parameters
    ----------
    obj : function object

    Returns
    -------
    decorated : function or instancemethod
    """
    cache = obj.cache = {}
    def memoizer(obj, *args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]

    decorated = decorator.decorator(memoizer, obj)
    return decorated

cached_function_fast = cached_function
cached_method = cached_function
