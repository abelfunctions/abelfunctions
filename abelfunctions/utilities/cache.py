r"""Cache :mod:`abelfunctions.utilities.cache`
==========================================

Code for cacheing functions.

Authors:

- Chris Swierczewski (November 2012)
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
    obj : object

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

    decorated = decorator.decorator(memoizer, func=obj)
    return decorated

cached_function_fast = cached_function
cached_method = cached_function
