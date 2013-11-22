'''
smale.pyx

NOTE:

* for some reason UnivariatePolynomial.eval is calling a bunch of
  Python. Run

      $ cython -a smale.pyx

  for more information
'''
cimport cython
from cython.view cimport array as cvarray
import numpy
cimport numpy

cdef extern from "math.h":
    double sqrt(double)

cdef extern from "complex.h":
    double creal(complex)
    double cimag(complex)
    double cabs(complex)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class UnivariatePolynomial:
    '''Fast(er) univariate polynomial evaluation.

    Converts a Sympy univariate polynomial to a faster univariate
    polynomial. Used by MultivariatePolynomial.

    Note: coefficients are stored in self.c in "reverse order". That is,
    the polynomial is

        c[0]*x^(deg) + c[1]*x^(deg-1) + ... + c[deg]
    '''
    def __cinit__(self,f,x):
        '''Initialize a UnivariatePolynomial from a Sympy Poly.'''
        cdef int n
        f = f.as_poly(x)
        coeffs = numpy.array(f.all_coeffs(),dtype=complex)
        self.deg = <int>(len(coeffs)-1)
        self.c = coeffs

    def __str__(self):
        cdef int n
        s = ''

        # special case for degree zero polynomials
        if (self.deg == 0):
            s = str(numpy.complex(self.c[0]))

        # add each nonzero term to the string
        for n in range(self.deg+1):
            cn = numpy.complex(self.c[n])
            # only add the last term if it's non-zero. otherwise, get
            # rid of the trailing plus sign
            if n == self.deg:
                if cn == 0.0:
                    m = len(s) # to avoid wraparound warning
                    s = s[:(m-3)]
                else:
                    s += str(cn)
            elif cn != 0.0:
                s += '%s*x**%d + '%(cn,self.deg-n)
        return s

    cdef complex eval(self,complex z) nogil:
        cdef complex acc = self.c[0]
        cdef int n
        for n in range(1,self.deg+1):
            acc = acc*z + self.c[n]
        return acc


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class MultivariatePolynomial:
    '''A Multivariate polynomial class for fast evaluation.'''
    def __cinit__(self,f,x,y):
        '''Initialize BiPolynomial from a Sympy polynomial.'''
        cdef int n

        f = f.as_poly(y)
        coeffs = f.all_coeffs()
        self.deg = <int>(len(coeffs)-1)
        self.c = numpy.array(
            [UnivariatePolynomial(coeff,x) for coeff in coeffs],
            dtype=UnivariatePolynomial)

    def __str__(self):
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
        cdef UnivariatePolynomial cn = self.c[0]
        cdef complex acc = cn.eval(z1)
        cdef int n
        for n in range(1,self.deg+1):
            cn = self.c[n]
            acc = acc*z2 + cn.eval(z1)
        return acc

    def is_zero_poly(self):
        if (self.deg == 0) and (self.c[0].is_zero_poly()):
            return True
        return False

cdef int factorial(int n) nogil:
    cdef int k, nfac = 1
    with nogil:
        for k in range(1,n+1):
            nfac *= k
        return nfac

@cython.boundscheck(False)
@cython.cdivision(True)
cdef complex newton(MultivariatePolynomial[:] df,
                    complex xip1,
                    complex yij):
    '''
    Newton iterate the y-root yij from the x-point xi to the x-point
    xip1.

    Input:

    - df: a list of all of the y-derivatives of f (up to the y-degree)

    - xip1: the x-point to analytically continue to

    - yij: a y-root at xi. The root that we'll analytically continue.
    '''
    cdef MultivariatePolynomial df0 = df[0]
    cdef MultivariatePolynomial df1 = df[1]
    cdef complex step = 1.0
    cdef complex df1y
    while cabs(step) > 1e-14:
        # if df is not invertible then we are at a critical point.
        df1y = df1.eval(xip1,yij)
        if cabs(df1y) < 1e-14:
            return yij

        step = df0.eval(xip1,yij)/df1y
        yij -= step
    return yij


@cython.boundscheck(False)
@cython.cdivision(True)
cdef double smale_beta(MultivariatePolynomial[:] df,
                       complex xip1,
                       complex yij):
    '''
    Compute Smale beta.

    Input:

    - df: a list of all of the y-derivatives of f (up to the y-degree)

    - xip1: the x-point to analytically continue to

    - yij: a y-root at xi. The root that we'll analytically continue.
    '''
    cdef MultivariatePolynomial df0 = df[0]
    cdef MultivariatePolynomial df1 = df[1]
    return cabs(df0.eval(xip1,yij)/df1.eval(xip1,yij))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double smale_gamma(MultivariatePolynomial[:] df,
                        complex xip1,
                        complex yij):
    '''
    Compute Smale gamma.

    Input:

    - df: a list of all of the y-derivatives of f (up to the y-degree)

    - xip1: the x-point to analytically continue to

    - yij: a y-root at xi. The root that we'll analytically continue.
    '''
    cdef MultivariatePolynomial df0 = df[0]
    cdef MultivariatePolynomial df1 = df[1]
    cdef MultivariatePolynomial dfn
    cdef int n, deg = df0.deg
    cdef complex df1y = df1.eval(xip1,yij)
    cdef double gamman, gamma = 0

    for n in range(2,deg+1):
        dfn = df[n]
        gamman = cabs(dfn.eval(xip1,yij) / (factorial(n)*df1y))
        gamman = gamman**(1.0/(n-1.0))
        if gamman > gamma:
            gamma = gamman
    return gamma


cdef double smale_alpha(MultivariatePolynomial[:] df,
                        complex xip1,
                        complex yij):
    '''Compute Smale gamma.

    Input:

    - dfs: a list of all of the y-derivatives of f (up to the y-degree)

    - xip1: the x-point to analytically continue to

    - yij: a y-root at xi. The root that we'll analytically continue.
    '''
    return smale_beta(df,xip1,yij) * smale_gamma(df,xip1,yij)
