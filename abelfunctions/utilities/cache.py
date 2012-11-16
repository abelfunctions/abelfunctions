"""
abelfunctions: cache.py

Code for cacheing functions.

Authors:

- Chris Swierczewski (November 2012)
"""

import collections
import functools

class cached_function(object):
    """
    Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args,**kwds):
        # if input is not cacheable then just call
        # the function (lest blow up)
        if not isinstance(args, collections.Hashable):
            return self.func(*args,**kwds)
       
        # check if in cache, otherwise, call function and
        # cache output
        key = (args, str(kwds))
        if key in self.cache:
            return self.cache[key]
        else:
            value = self.func(*args,**kwds)
            self.cache[key] = value
            return value
       
    def __repr__(self):
        """
        Return the function's docstring.
        """
        return self.func.__doc__
    
    def __get__(self, obj, objtype):
        """
        Support instance methods.
        """
        return functools.partial(self.__call__, obj)


if __name__=="__main__":
    @memoized
    def foo(a,n=5):
        return a*n

    print foo(1)
    print foo(2)
    print foo(1,n=6)
    print foo(2,6)

    print foo.cache
