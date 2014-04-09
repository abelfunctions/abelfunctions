"""
Polynomials
===========

Fast, Cython-level implementation of Univariate and Multivariate polynomials.

Authors
-------

* Chris Swierczewski (April 2014)
"""

import numpy
cimport numpy
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class UnivariatePolynomial:
    """Fast complex univariate polynomial.

    Converts a SymPy univariate polynomial to a faster univariate
    polynomial. Used by MultivariatePolynomial.

    .. note:: coefficients are stored in self.c in "reverse order". That
              is, the polynomial is given by

                  c[0]*x^(deg) + c[1]*x^(deg-1) + ... + c[deg]

    Attributes
    ----------
    deg : int
        The degree of the polynomial.
    c : int[:]
        The coefficients of the polynomial starting with the degree
        `deg` term and ending with the constant term.

    Methods
    -------
    eval(complex z)
        Evaluate the polynomial at the complex point `z`.
    """
    def __cinit__(self,f,x):
        """Initialize a UnivariatePolynomial from a SymPy Poly.

        Arguments
        ---------
        f : SymPy Poly
        x : SymPy symbol
            The polynomial and its independent varaible.
        """
        cdef int n
        f = f.as_poly(x)
        coeffs = numpy.array(f.all_coeffs(),dtype=complex)
        self.deg = len(coeffs) - 1
        self.c = coeffs

    def __repr__(self):
        cdef int n
        s = ''

        # special case for degree zero polynomials
        if (self.deg == 0):
            s = str(numpy.complex(self.c[0]))

        for n in range(self.deg+1):
            cn = numpy.complex(self.c[n])
            # only add the last term if it's non-zero. otherwise, get
            # rid of the trailing plus sign
            if n == self.deg:
                if cn == 0.0:
                    m = len(s)
                    s = s[:(m-3)]
                else:
                    s += str(cn)
            elif cn != 0.0:
                s += '%s*x**%d + '%(cn,self.deg-n)
        return s

    cdef complex eval(self, complex z) nogil:
        """Evaluate the polynomial at the complex point `z`.

        Arguments
        ---------
        z : complex

        Returns
        -------
        complex
            Evaluates the polynomial and returns :math:`f(z)`.
        """
        cdef complex acc = self.c[0]
        cdef int n
        for n in range(1,self.deg+1):
            acc = acc*z + self.c[n]
        return acc

    def is_zero_poly(self):
        if (self.deg == 0) and (self.c[0] == 0):
            return True
        return False


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class MultivariatePolynomial:
    """Fast complex multivariate polynomial.

    Converts a SymPy bivariate polynomial to a polynomial object that
    performs fast complex evaluation.

    .. note:: coefficients are stored in self.c in "reverse order". That
              is, the polynomial is given by

                  c[0](x)*y^(deg) + c[1](x)*y^(deg-1) + ... + c[deg](x)

    Attributes
    ----------
    deg : int
        The degree of the polynomial.
    c : UnivariatePolynomial[:]
        The coefficients of the polynomial starting with the degree
        `deg` term and ending with the constant term.

    Methods
    -------
    eval(complex z)
        Evaluate the polynomial at the complex point `z`.
    """
    def __cinit__(self,f,x,y):
        """Initialize a MultivariatePolynomial from a SymPy Poly.

        Arguments
        ---------
        f : SymPy Poly
        x : SymPy symbol
        y : SymPy symbol
            The polynomial and its variables. Note that multivariate
            polynomials are considered to be polynomials in `y` whose
            coefficients are polynomials in `x`.
        """
        cdef int n
        f = f.as_poly(y)
        coeffs = f.all_coeffs()
        self.deg = len(coeffs) - 1
        self.c = numpy.array(
            [UnivariatePolynomial(coeff,x) for coeff in coeffs],
            dtype=UnivariatePolynomial)

    def __repr__(self):
        cdef int n
        s = ''
        for n in range(self.deg+1):
            if n == self.deg:
                if self.c[n].is_zero_poly():
                    m = len(s)
                    s = s[:(m-3)]
                else:
                    s += self.c[self.deg].__str__()
            elif not self.c[n].is_zero_poly():
                s += '(%s)y**%d + '%(self.c[n].__str__(),self.deg-n)
        return s

    cdef complex eval(self,complex z1,complex z2):
        """Evaluate the polynomial at the complex point `z1,z2`.

        Arguments
        ---------
        z1,z2 : complex


        Returns
        -------
        complex
            Evaluates the polynomial and returns :math:`f(z1,z2)`.
        """
        cdef UnivariatePolynomial cn = self.c[0]
        cdef complex acc = cn.eval(z1)
        cdef int n
        for n in range(1,self.deg+1):
            cn = self.c[n]
            acc = acc*z2 + cn.eval(z1)
        return acc

    def is_zero_poly(self):
        """Returns `True` if the polynomial is the zero polynomial."""
        if (self.deg == 0) and (self.c[0].is_zero_poly()):
            return True
        return False
