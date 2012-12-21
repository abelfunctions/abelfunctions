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

import time

class cached_property(object):
    """
    Decorator for read-only properties evaluated only once within TTL period.

    It can be used to created a cached property like this::

        import random

        # the class containing the property must be a new-style class
        class MyClass(object):
            # create property whose value is cached for ten minutes
            @cached_property(ttl=600)
            def randint(self):
                # will only be evaluated every 10 min. at maximum.
                return random.randint(0, 100)

    The value is cached  in the '_cache' attribute of the object instance that
    has the property getter method wrapped by this decorator. The '_cache'
    attribute value is a dictionary which has a key for every property of the
    object which is wrapped by this decorator. Each entry in the cache is
    created only when the property is accessed for the first time and is a
    two-element tuple with the last computed property value and the last time
    it was updated in seconds since the epoch.

    The default time-to-live (TTL) is 300 seconds (5 minutes). Set the TTL to
    zero for the cached value to never expire.

    To expire a cached property value manually just do::

        del instance._cache[<property name>]

    """
    def __init__(self, ttl=0):
        self.ttl = ttl

    def __call__(self, fget, doc=None):
        self.fget = fget
        self.__doc__ = doc or fget.__doc__
        self.__name__ = fget.__name__
        self.__module__ = fget.__module__
        return self

    def __get__(self, inst, owner):
        now = time.time()
        try:
            value, last_update = inst._cache[self.__name__]
            if self.ttl > 0 and now - last_update > self.ttl:
                raise AttributeError
        except (KeyError, AttributeError):
            value = self.fget(inst)
            try:
                cache = inst._cache
            except AttributeError:
                cache = inst._cache = {}
            cache[self.__name__] = (value, now)
        return value


if __name__=="__main__":
    @cached_function
    def foo(a,n=5):
        return a*n

    print foo(1)
    print foo(2)
    print foo(1,n=6)
    print foo(2,6)

    print foo.cache
