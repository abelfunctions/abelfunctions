#cython: embedsignature=True
r"""Riemann Theta Function :mod:`abelfunctions.riemann_theta.riemann_theta`
=======================================================================

The primary module for computing the Riemann theta function.

.. math::

    \theta(z, \Omega) = \sum_{n \in \mathbb{Z}^g}
                        e^{2 \pi i \left( \tfrac{1}{2} n \cdot \Omega n
                           + n \cdot z \right)}

Functions
---------

oscillatory_part

Classes
-------

RiemannTheta_Function

Contents
--------

"""
cimport cython
import numpy

from libc.stdlib cimport malloc
from abelfunctions.riemann_theta.radius import radius
from abelfunctions.riemann_theta.integer_points import integer_points_python

cdef extern from *:
    void finite_sum_without_derivatives(double*, double*, double*,
                                        double*, double*, double*,
                                        double*, double*, int, int)
    void finite_sum_with_derivatives(double*, double*, double*, double*,
                                     double*, double*, double*, double*,
                                     double*, double*, int, int, int)

@cython.boundscheck(False)
@cython.wraparound(False)
def oscillatory_part(z, Omega, epsilon, derivs, accuracy_radius, axis):
    r"""Compute the oscillatory part of the Riemann theta function.

    See :func:`RiemannTheta_Function.oscillatory_part` for information on the
    arguments.
    """
    cdef double[:] x
    cdef double[:] y
    cdef int g
    cdef int k
    cdef int num_vectors
    cdef int N
    cdef double[:,:] X
    cdef double[:,:] Yinv
    cdef double[:,:] T
    cdef double[:,:] S
    cdef double* real
    cdef double* imag
    cdef double[:] derivs_real
    cdef double[:] derivs_imag
    cdef int nderivs

    # coerce z to a numpy array and determine the problem size: the genus g and
    # the number of vectors to compute
    z = numpy.array(z, dtype=numpy.complex)
    if len(z.shape) == 1:
        num_vectors = 1
        g = len(z)
    else:
        num_vectors = z.shape[abs(axis-1)]
        g = z.shape[axis]
    z = z.flatten()

    # coerce Omega to a numpy array and extract the requested information: the
    # real part, inverse of the imaginary part, and the cholesky decomposition
    # of the imaginary part
    Omega = numpy.array(Omega, dtype=numpy.complex)
    Y = Omega.imag
    _T = numpy.linalg.cholesky(Y).T
    X = numpy.ascontiguousarray(Omega.real)
    Yinv = numpy.ascontiguousarray(numpy.linalg.inv(Y))
    T = numpy.ascontiguousarray(_T)

    # compute the integer points over which we approximate the infinite sum to
    # the requested accuracy
    R = radius(epsilon, _T, derivs=derivs, accuracy_radius=accuracy_radius)
    S = numpy.ascontiguousarray(integer_points_python(g,R,_T))
    N = S.shape[0]

    # set up storage locations and vectors
    real = <double*>malloc(sizeof(double))
    imag = <double*>malloc(sizeof(double))
    values = numpy.zeros(num_vectors, dtype=numpy.complex)

    # get the derivatives
    if len(derivs):
        derivs = numpy.array(derivs, dtype=numpy.complex).flatten()
        nderivs = len(derivs) / g
        derivs_real = numpy.ascontiguousarray(derivs.real, dtype=numpy.double)
        derivs_imag = numpy.ascontiguousarray(derivs.imag, dtype=numpy.double)

        # compute the finite sum for each z-vector
        for k in range(num_vectors):
            zk = z[k*g:(k+1)*g]
            x = numpy.ascontiguousarray(zk.real, dtype=numpy.double)
            y = numpy.ascontiguousarray(zk.imag, dtype=numpy.double)
            finite_sum_with_derivatives(real, imag,
                                        &X[0,0], &Yinv[0,0], &T[0,0],
                                        &x[0], &y[0], &S[0,0],
                                        &derivs_real[0], &derivs_imag[0],
                                        nderivs, g, N)
            value = numpy.complex(real[0] + 1.0j*imag[0])
            values[k] = value
    else:
        # compute the finite sum for each z-vector
        for k in range(num_vectors):
            zk = z[k*g:(k+1)*g]
            x = numpy.ascontiguousarray(zk.real, dtype=numpy.double)
            y = numpy.ascontiguousarray(zk.imag, dtype=numpy.double)
            finite_sum_without_derivatives(real, imag,
                                           &X[0,0], &Yinv[0,0], &T[0,0],
                                           &x[0], &y[0], &S[0,0], g, N)
            value = numpy.complex(real[0] + 1.0j*imag[0])
            values[k] = value

    if num_vectors == 1:
        return values[0]
    else:
        return values


cdef class RiemannTheta_Function(object):
    r"""The Riemann theta function.

    This class is globally instantiated as `RiemannTheta`.

    Methods
    -------
    eval
    exponential_part
    oscillatory_part

    """
    def __init__(self, accuracy_radius=5):
        pass

    def __call__(self, *args, **kwds):
        r"""Returns the value of the Riemann theta function at `z` and `Omega`.

        See :meth:`eval` for documentation.
        """
        return self.eval(*args, **kwds)

    def eval(self, z, Omega, **kwds):
        r"""Returns the value of the Riemann theta function at `z` and `Omega`.

        In many applications it's preferred to use :meth:`exponential_part` and
        :meth:`oscillatory_part` due to the double-exponential growth of theta
        in the directions of the columns of `Omega`.

        Parameters
        ----------
        z : complex[:]
            A complex row-vector or list of row-vectors at which to evaulate the
            Riemann theta function.
        Omega : complex[:,:]
            A Riemann matrix.
        **kwds : keywords
            See :meth:`exponential_part` and :meth:`oscillatory_part` for
            optional keywords.

        Returns
        -------
        array
            The value of the Riemann theta function at each `g`-component vector
            appearing in `z`.
        """
        u = self.exponential_part(z, Omega, **kwds)
        v = self.oscillatory_part(z, Omega, **kwds)
        values = numpy.exp(u)*v
        return values

    def exponential_part(self, z, Omega, axis=1, **kwds):
        r"""Returns the exponential part of the Riemann theta function.

        This function is "vectorized" over `z`. By default, each row of `z` is
        interpreted as a separate input vector to the Riemann theta function.

        Parameters
        ----------
        z : complex[:]
            A complex row-vector or list of row-vectors at which to evaulate the
            Riemann theta function.
        Omega : complex[:,:]
            A Riemann matrix.
        axis : int
            (Default: `1`) If multiple `z`-vectors are given in the form of a
            two-dimensional array, specify over which axis to compute the
            Riemann theta function. By default, each row of `z` is interpreted
            as an input vector.

        Returns
        -------
        array
            The value of the exponential part of the Riemann theta function at
            each `g`-component vector appearing in `z`.

        """
        # extract the imaginary parts of z and the inverse of the imaginary part
        # of Omega
        if len(numpy.shape(z)) == 1:
            z = [z]
        z = numpy.array(z, dtype=numpy.complex)
        Omega = numpy.array(Omega, dtype=numpy.complex)
        y = z.imag
        Yinv = numpy.linalg.inv(Omega.imag)

        # apply the quadratic form to each vector in z
        quad = lambda yi: numpy.pi*numpy.dot(yi,numpy.dot(Yinv,yi))
        exponents = numpy.apply_along_axis(quad, axis, y)
        if len(exponents) == 1:
            return exponents[0]
        else:
            return exponents


    def oscillatory_part(self, z, Omega, epsilon=1e-8, derivs=[],
                         accuracy_radius=5., axis=1, **kwds):
        r"""Compute the oscillatory part of the Riemann theta function.

        The oscillatory part of the Riemann theta function is the infinite
        summation left over after factoring out the double-exponential growth.

        This function is "vectorized" over `z` in order to take advantage of the
        uniform approximation theorem.

        Parameters
        ----------
        z : complex[:]
            A complex row-vector or list of row-vectors at which to evaluate the
            Riemann theta function.
        Omega : complex[:,:]
            A Riemann matrix.
        epsilon : double
            (Default: `1e-8`) The desired numerical accuracy.
        derivs : list of lists
            (Default: `[]`) A directional derivative given as a list of lists.
        accuracy_radius : double
            (Default: `5.`) The raidus from the g-dimensional origin where the
            requested accuracy of the Riemann theta is guaranteed when computing
            derivatives. Not used if no derivatives of theta are requested.
        axis : int
            (Default: `1`) If multiple `z`-vectors are given in the form of a
            two-dimensional array, specify over which axis to compute the
            Riemann theta function. By default, each row of `z` is interpreted
            as an input vector.

        Returns
        -------
        array
            The value of the Riemann theta function at each `g`-component vector
            appearing in `z`.

        """
        return oscillatory_part(z, Omega, epsilon, derivs,
                                accuracy_radius, axis)

# declaration of Riemann theta
RiemannTheta = RiemannTheta_Function()
