r"""Riemann Theta Function :mod:`abelfunctions.riemann_theta.riemann_theta`
=======================================================================

The primary module for computing the Riemann theta function.

.. math::

    \theta(z, \Omega) = \sum_{n \in \mathbb{Z}^g}
                        e^{2 \pi i \left( \tfrac{1}{2} n \cdot \Omega n
                           + n \cdot z \right)}

Classes
-------

.. autosummary::

    RiemannTheta_Function

Functions
---------

.. autosummary::

    oscillatory_part

References
----------

.. [CRTF] B. Deconinck, M.  Heil, A. Bobenko, M. van Hoeij and M. Schmies,
   Computing Riemann Theta Functions, Mathematics of Computation, 73, (2004),
   1417-1442.

.. [DLMF] B. Deconinck, Digital Library of Mathematics Functions - Riemann
   Theta Functions, http://dlmf.nist.gov/21

.. [SAGE] Computing Riemann theta functions in Sage with applications.
   C. Swierczewski and B. Deconinck.Submitted for publication.  Available
   online at
   http://depts.washington.edu/bdecon/papers/pdfs/Swierczewski_Deconinck1.pdf


Examples
--------

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

    Attributes
    ----------
    none

    Methods
    -------
    exponential_part
    oscillatory_part
    oscillatory_part_gradient
    gradient
    oscillatory_part_hessian
    eval

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

    def oscillatory_part_gradient(self, z, Omega, epsilon=1e-8,
                                  accuracy_radius=5, axis=1, **kwds):
        r"""Returns the oscillatory part of the gradient of Riemann theta.

        A helper function for :meth:`gradient`. Useful in it's own right for
        detecting if the gradient vanishes since the exponential part of
        Riemann theta never does as well as for controlling exponential growth in
        rational functions of theta.

        Parameters
        ----------
        (see :meth:`oscillatory_part`)

        Returns
        -------
        array
            If a single z-argument is given then returns a 1-dimensional Numpy
            array representing the gradient. If multiple z-arguments are given
            then returns a 2-dimensional Numpy array of gradients where each
            gradient is listed along `axis`.
        """
        # get the genus and number of z-vectors
        Omega = numpy.array(Omega, dtype=numpy.complex)
        g = Omega.shape[0]
        gradients = numpy.zeros_like(z, dtype=numpy.complex)

        for i in range(g):
            # construct the direction derivative \partial z_i and compute the
            # derivatives in that direction
            derivs = [[0]*g]
            derivs[0][i] = 1
            partial_zi = oscillatory_part(z, Omega, epsilon, derivs,
                                          accuracy_radius, axis)

            # if axis=1 then store gradients in rows (since input z-vectors are
            # given as rows of a matrix). otherwise, store column-wise
            if axis:
                gradients[:,i] = partial_zi
            else:
                gradients[i,:] = partial_zi

        return gradients

    def gradient(self, z, Omega, epsilon=1e-8, accuracy_radius=5,
                 axis=1, **kwds):
        r"""Returns the gradient of Riemann theta.

        The gradient of :math:`\theta((z_1, \ldots, z_g), \Omega)` is defined
        to be

        .. math::

            \grad \theta(z,\Omega) = \left( \ldots, \partial_{z_i} \theta(z,\Omega), \ldots \right)

        where :math:`\partial_{z_i}` denotes the directional derivative in the
        :math:`[\ldots,0,1,0,\ldots]` direction. (The "1" occurs in the
        :math:`i` th position.)

        See Also
        --------
        oscillatory_part_gradient

        Parameters
        ----------
        (see :meth:`oscillatory_part`)

        Returns
        -------
        array
            If a single z-argument is given then returns a 1-dimensional Numpy
            array representing the gradient. If multiple z-arguments are given
            then returns a 2-dimensional Numpy array of gradients where each
            gradient is listed along `axis`.

        Notes
        -----
        This could be made shorter by using :meth:`eval` but we choose not to
        for performance reasons. (We avoid having to compute the exponential
        part multiple times.)

        """
        # compute the gradient of the oscillatory part, the exponential part,
        # and combine "axis-wise": if the z-vectors are given as row vectors
        # then multiply each column of the gradient of the oscillatory part by
        # the exponential part
        osc_gradients = self.oscillatory_part_gradient(
            z, Omega, epsilon, accuracy_radius, axis, **kwds)
        exp_parts = self.exponential_part(z, Omega, **kwds)
        exp_parts = numpy.exp(exp_parts)
        gradients = numpy.apply_along_axis(
            lambda dz: exp_parts*dz,
            (axis+1)%2,
            osc_gradients)
        return gradients

    def oscillatory_part_hessian(self, z, Omega, epsilon=1e-8,
                                 accuracy_radius=5, axis=1, **kwds):
        r"""Returns the oscillatory part of the Hessian of Riemann theta.

        A helper function for :meth:`hessian`.  Useful in it's own right for
        detecting if the Hessian vanishes since the exponential part of Riemann
        theta never does as well as for controlling exponential growth in
        rational functions of theta.

        Parameters
        ----------
        (see :meth:`oscillatory_part`)

        Returns
        -------
        array
            If a single z-argument is givne then returns a 2-dimensional Numpy
            array representing the Hessian. If multiple z-arguments are given
            then returns a 3-dimensional Numpy array of Hessians where each
            Hessian is indexed by the 0th coordinate.
        """
        # get the genus and number of z-vectors
        Omega = numpy.array(Omega, dtype=numpy.complex)
        g = Omega.shape[0]
        z = numpy.array(z, dtype=numpy.complex)
        if len(z.shape) == 1:
            n = 1
        else:
            n = z.shape[(axis+1) % 2]
        hessians = numpy.zeros((n,g,g), dtype=numpy.complex)

        # since Riemann theta is analytic we only need to compute the lower
        # triangular portion of the Hessian and symmetrize
        lower_derivs = []
        for i in range(g):
            d1 = [0]*g
            d1[i] = 1
            for j in range(i):
                d2 = [0]*g
                d2[j] = 1

                # construct the derivative and evaluate the corresponding zi zj
                # derivative across all input vectors
                derivs = [d1, d2]
                partial_zizj = oscillatory_part(z, Omega, epsilon, derivs,
                                                accuracy_radius, axis)
                hessians[:,i,j] = partial_zizj

        # symmetrize
        hessians += hessians.transpose((0,2,1))

        # finally, compute the diagonal entries of each hessian
        for i in range(g):
            d1 = [0]*g
            d1[i] = 1
            derivs = [d1,d1]
            partial_zizi = oscillatory_part(z, Omega, epsilon, derivs,
                                            accuracy_radius, axis)
            hessians[:,i,i] = partial_zizi

        if n == 1:
            return hessians[0,:,:]
        else:
            return hessians


# declaration of Riemann theta
RiemannTheta = RiemannTheta_Function()
