"""
Permutations
============

A simple implementation of permutations on `n` elements.

Authors
-------

* Chris Swierczewski (Feb 2014)
"""

class Permutation(object):
    """A permutation on `n` elements.

    Methods
    -------
    is_identity()
        Returns `True` if the Permutation is the identity.
    index(j)
        Representing the Permutation in "map" notation, a list where `i`
        is mapped to `j = lst[i]`, returns `i`. That is, the preimage of
        `j`.
    action(a)
        Returns the permutation of an iterable `a` under the action of
        the permutation.
    inverse()
        Returns the inverse of the Permutation.
    """
    def __init__(self, l):
        """Construct a Permutation from a list.

        There are two ways to constuct a permutation.

        1. Permutations can be initialized by a list which is a
           permutation of `range(n)` given in "map" notation. That is,
           given a list `lst` the permutation constructed maps `i` to
           `lst[i]`.

        2. Permutations can be initialized by a list representing the
           permutation in cycle notation. Fixed cycles must be provided.

        Parameters
        ----------
        l : iterable
            Either an iterable (list) of integers from `0` to `n-1` or
            an iterable of iterables.

        Examples
        --------
        We construct the permutation `p = 0->3, 1->1, 2->0, 3->2` in two
        different ways. First, we construct the permutation from a "map".

        >>> p = Permutation([3,1,0,2])
        >>> print(p)
        foo

        Second, the same permutation in cycle notation.

        >>> q = Permutation([[1], [0,3,2]])
        >>> print(q)
        foo
        >>> p == q
        True
        """
        if isinstance(l,list):
            if isinstance(l[0],list):
                l = self._list_from_cycles(l)
            self._list = l
        else:
            # try to turn object into list
            self._list = list(l)
            self.__init__(l)

        self._cycles = self._cycles_from_list(self._list)
        self._hash = None


    def _list_from_cycles(self, cycles):
        """Create a permutation list `i \to l[i]` from a cycle notation list.

        Examples
        --------
        >>> p = Permutation([[0,1],[2],[3]])
        >>> p._list
        [1, 0, 2]

        >>> q = Permutation([[2,4],[1,3]])
        >>> q._list
        [2, 3, 0, 1]
        """
        degree = max([0] + [max(cycle + [0]) for cycle in cycles]) + 1
        l = list(range(degree))
        for cycle in cycles:
            if not cycle:
                continue
            first = cycle[0]
            for i in range(len(cycle)-1):
                l[cycle[i]] = cycle[i+1]
            l[cycle[-1]] = first

        return l

    def _cycles_from_list(self,l):
        """Create a list of cycles from a permutation list."""
        n = len(l)
        cycles = []
        not_visited = list(range(n))[::-1]

        while len(not_visited) > 0:
            i = not_visited.pop()
            cycle = [i]
            j = l[i]
            while j != i:
                cycle.append(j)
                not_visited.remove(j)
                j = self(j)
            cycles.append(tuple(cycle))

        return cycles

    def __repr__(self):
        non_identity_cycles = [c for c in self._cycles if len(c) > 1]
        return str(non_identity_cycles)

    def __hash__(self):
        if self._hash is None:
            self._hash = str(self._list).__hash__()
        return self._hash

    def __len__(self):
        return self._list.__len__()

    def __getitem__(self, key):
        return self._list.__getitem__(key)

    def __contains__(self, item):
        return self._list.__contains__(item)

    def __eq__(self, other):
        return self._list == other._list

    def __mul__(self, other):
        return self.__rmul__(other)

    def __rmul__(self, other):
#         # pad the permutations if they are of different lengths
#         new_other = other[:] + [i+1 for i in range(len(other), len(self))]
#         new_p1 = self[:] + [i+1 for i in range(len(self), len(other))]
#         return Permutation([new_p1[i-1] for i in new_other])
        new_other = other[:] + [i for i in range(len(other), len(self))]
        new_p1 = self[:] + [i for i in range(len(self), len(other))]
        return Permutation([new_p1[i] for i in new_other])

    def __call__(self, i):
        """Returns the image of the integer i under this permutation."""
        if isinstance(i,int) and 0 <= i < len(self):
            return self[i]
        else:
            raise TypeError("i (= %s) must be an integer between "
                            "%s and %s" % (i, 0, len(self) - 1))

    def is_identity(self):
        """Returns `True` if permutation is the identity."""
        n = len(self._list)
        return self._list == list(range(n))

    def index(self, j):
        """If `p(i) = j`, returns `i`."""
        return self._list.index(j)

    def action(self, a):
        """Returns the action of the permutation on an iterable.

        Examples
        --------
        >>> p = Permutation([0,3,1,2])
        >>> p.action(['a','b','c','d'])
        ['a', 'd', 'b', 'c']
        """
        if len(a) != len(self):
            raise ValueError("len(a) must equal len(self)")
        return [a[self[i]] for i in range(len(a))]

    def inverse(self):
        """
        Returns the inverse permutation.
        """
        l = list(range(len(self)))
        for i in range(len(self)):
            l[self(i)] = i

        return Permutation(l)


def matching_permutation(a, b):
    """Returns the permutation `p` mapping the elements of `a` to the
    elements of `b`.

    This function returns a :class:`Permutation` `p` such that `b ~
    p.action(a)` or, equivalently, `norm(b - p.action(a))` is small. The
    elements of `a` and `b` need not be exactly the same but close
    enough to each other that it's unambiguous which elements match.

    Parameters
    ----------
    a,b : iterable
        Lists of approximately the same elements.

    Returns
    -------
    Permutation
        A Permutation `p` such that `norm(b - p.action(a))` is small.

    Examples
    --------

    If the two lists contain the same elements then
    `matching_permutation` simply returns permutation defining the
    rearrangement.

    >>> a = [6, -5, 9]
    >>> b = [9, 6, -5]
    >>> p = matching_permutation(a,b); p
    [2, 0, 1]

    `matching_permutation` will attempt to find such a permutation even
    if the elements of the two lists are not exactly the same.

    >>> a = [1.1, 7.2, -3.9]
    >>> b = [-4, 1, 7]
    >>> p = matching_permutation(a,b); p
    [2, 0, 1]
    >>> p.action(a)
    [-3.9, 1.1, 7.2]
    """
    N = len(a)
    if N != len(b):
        raise ValueError("Lists must be of same length.")

    perm = [-1]*N
    eps  = 0.5*min([abs(a[i]-a[j]) for i in range(N) for j in range(i)])

    for i in range(N):
        for j in range(N):
            dist = abs(a[i] - b[j])
            if dist < eps:
                perm[j] = i
                break

    if -1 in perm:
        raise ValueError("Could not compute matching permutation "
                         "between %s and %s." % (a, b))

    return Permutation(perm)



