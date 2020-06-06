r""" Places and Divisors :mod:`divisor`
==================================

A module defining places and divisors on a Riemann surface. A *regular
affine place* is simply given as an :math:`(x_0,y_0)` tuple on the curve
:math:`C : f(x,y) = 0` from which the Riemann surface is defined.

Classes
-------

.. autosummary::

    Divisor
    Place


Functions
---------

.. autosummary::

    ZeroDivisor


Examples
--------

Contents
--------

"""

from sage.all import infinity


class Divisor(object):
    r"""A divisor on a Riemann surface.

    A divisor is a formal sum of :py:class:`Place`s :math:`P_i` along
    with multiplicities :math:`m_i` usually written,

    .. math::

        D = m_1 P_1 + \cdots m_k P_k,

    where the places :math:`P_i` are distinct.

    Attributes
    ----------
    places : list
        A list of the distinct places in this divisor.
    multiplicities : list
        The corresponding multiplicities of the places in this divisor.

    """
    @property
    def places(self):
        return tuple(self._d.keys())

    @property
    def multiplicities(self):
        return tuple(self._d.values())

    @property
    def items(self):
        return tuple(self._d.items())

    @property
    def degree(self):
        return sum(self.multiplicities)

    @property
    def dict(self):
        return self._d

    @dict.setter
    def dict(self, value):
        d = {P: m for P, m in value.items() if m}
        self._d = d

    def __init__(self, RS, d):
        r"""Initialize a Divisor.

        Divisors can be constructed from a place, an iterable of places,
        or a dict with places as keys and multiplicities as values.

        Parameters
        ----------
        RS : RiemannSurface
            The surface on which the place is defined.
        d : dict
            A dictionary of places as keys and multiplicities as values.
        """
        self.RS = RS
        if isinstance(d, Divisor):
            self.__dict__ = d.__dict__.copy()
        elif isinstance(d, dict):
            self.dict = d
        elif d == 0:
            # construct the zero divisor
            self.dict = {}
        else:
            raise ValueError('d must be a dictionary')

    def __repr__(self):
        if not self.is_zero():
            s = ''
            for P,n in self:
                s += ' + '
                if n > 1:
                    s += str(n)
                elif n < 0:
                    s += '(' + str(n) + ')'
                s += P.name
            return s[3:]
        else:
            return 'Div0'

    def __getitem__(self, key):
        try:
            return self.dict[key]
        except KeyError:
            return 0

    def __iter__(self):
        return iter(self.dict.items())

    def __key(self):
        return (self.RS, self.dict)

    def __eq__(self, other):
        if isinstance(other, Divisor):
            return self.__key() == other.__key()
        return False

    def __hash__(self):
        # Unable to use dict in hash since dict is mutable
        return hash(self.RS)

    def __add__(self, other):
        if other == 0:
            return self

        if not isinstance(other,Divisor):
            raise ValueError('%s is not a Divisor on %s.'%(other,self.RS))

        if self.RS != other.RS:
            raise ValueError('Can only add or subtract divisors defined '
                             'on the same Riemann surface.')

        all_places = set(self.places + other.places)
        d = dict((P, self[P] + other[P]) for P in all_places)
        return Divisor(self.RS, d)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        d = dict((P,-m) for P,m in self.items)
        return Divisor(self.RS, d)

    def __sub__(self, other):
        return self + other.__neg__()

    def __mul__(self, other):
        other = int(other)
        d = dict((P,other*m) for P,m in self.items)
        return Divisor(self.RS, d)

    def __rmul__(self, other):
        return self.__mul__(other)

    def is_zero(self):
        if self.dict == {}:
            return True
        return False

    def as_place(self):
        items = list(self.dict.items())
        if len(items) != 1:
            raise ValueError('Divisor contains more than one place. '
                             'Cannot coerce to place.')
        P, m = items[0]
        if m != 1:
            raise ValueError('Divisor contains place of multiplicity. '
                             'Cannot coerce to single place.')
        return P


class ZeroDivisor(Divisor):
    r"""A class representing the zero divisor on a Riemann surface."""
    def __init__(self, RS):
        Divisor.__init__(self, RS, {})


class Place(Divisor):
    r"""A Place on a Riemann surface.

    In `abelfunctions` a Riemann surface is obtained by desingularizing
    and compactifying a complex algebraic curve. Every place :math:`P`
    on the resulting Riemann surface can be described in terms of a
    Puiseux series in a local parameter :math:`t`.

    .. math::

        P = (x(t), y(t))
        x(t) = \alpha + \lambda t^e
        y(t) = \sum_k \beta_k t^{n_k}

    When :math:`\alpha` is not a discriminant point or infinity of the
    curve then :math:`P = (x(0), y(0)) = (\alpha, \beta)` is sufficient
    in representing the place from the curve. Otherwise, multiple places
    may have the same projection onto the curve.

    Attributes
    ----------
    x : complex
        The x-projection of the place onto the underlying curve.
    y : complex
        The y-projection of the palce onto the underlying curve.
    name : string, optional
        A name given to the place. Used when the place is a term in a
        :py:class:`Divisor`.

    Methods
    -------

    is_discriminant

    """
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not value:
            value = str((self.x,self.y))
        self._name = value

    def __init__(self, RS, name=None):
        self.RS = RS
        self.name = name
        Divisor.__init__(self, RS, {self:1})

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        raise NotImplementedError('Override in Place subtype.')

    def __hash__(self):
        raise NotImplementedError('Override in Place subtype.')

    def as_place(self):
        return self

    def is_discriminant(self):
        raise NotImplementedError('Override in Place subtype.')

    def is_infinite(self):
        raise NotImplementedError('Override in Place subtype.')

    def valuation(self, omega):
        r"""Returns the valuation of `omega` at this place.

        The valuation of :math:`\omega` at this place :math:`P` is an
        integer :math:`m` such that

        .. math::

            \omega |_P = c t^m + O(t^{m+1})

        where :math:`P = P(0)` is given in terms of some local parameter
        :math:`t`.

        This method is a key ingredient in determining the valuation
        divisor of a differential :math:`\omega`.

        Parameters
        ----------
        omega : Differential

        Returns
        -------
        int
        """
        raise NotImplementedError('Override in Place subtype')


class RegularPlace(Place):
    r"""A regular place on a Riemann surface.

    """
    def __init__(self, RS, x, y, **kwds):
        r"""Initialize a regular place from a point on the curve.

        Parameters
        ----------
        RS : RiemannSurface
        x : complex
            The x-projection of the place onto the curve.
        y : complex
            The y-projection of the place onto the curve.
        """
        self.x = x
        self.y = y
        Place.__init__(self, RS, **kwds)

    def __key(self):
        return (self.RS, self.x, self.y)

    def __eq__(self, other):
        if isinstance(other, RegularPlace):
            return self.__key() == other.__key()
        return False

    def __hash__(self):
        return hash(self.__key())

    def is_discriminant(self):
        return False

    def is_infinite(self):
        return False

    def valuation(self, omega):
        f = self.RS.f
        x,y = f.parent().gens()
        a,b = self.x, self.y

        def mult(p, a, b):
            r"""Returns the multiplicity of the zero `(a,b)` on `p`."""
            pab = p(x + a, y + b)
            degrees = [sum(exp) for exp in pab.exponents()]
            return min(degrees)

        numer,denom = omega.as_numer_denom() # XXX
        zero_mult = mult(numer,a,b)
        pole_mult = mult(denom,a,b)
        return zero_mult - pole_mult


class DiscriminantPlace(Place):
    r"""A discriminant place on a Riemann surface.
    """
    def __init__(self, RS, P, **kwds):
        r"""Initialize a disc. place from its Puiseux series representation.

        Parameters
        ----------
        RS : RiemannSurface
        P : PuiseuxTSeries
        """
        self.puiseux_series = P
        self.x = P.x0
        self.y = P.eval_y(0)
        Place.__init__(self, RS, **kwds)
        self.name = str(P)

    def __repr__(self):
        return str(self.puiseux_series)

    def __key(self):
        return (self.RS, self.puiseux_series)

    def __eq__(self, other):
        if isinstance(other, DiscriminantPlace):
            return self.__key() == other.__key()
        return False

    def __hash__(self):
        return hash(self.__key())

    def is_discriminant(self):
        return True

    def is_infinite(self):
        xval = self.puiseux_series.eval_x(0)
        if xval == infinity:
            return True
        return False

    def valuation(self, omega):
        omegat = omega.localize(self)
        valuation = omegat.valuation()
        return valuation

