cimport cython
import numpy
cimport numpy


cdef extern from 'finite_sum.h':
    void finite_sum_without_derivatives(double*, double*, double*,
                                        double*, double*, double*,
                                        double*, double*, int, int)
    void finite_sum_with_derivatives(double*, double*, double*, double*,
                                     double*, double*, double*, double*,
                                     double*, double*, int , int , int )


cpdef complex finite_sum(double[:,:] X, double[:,:] Yinv, double[:,:] T,
                         double[:] x, double[:] y, int[:,:] S, int g, int N):
    r"""Compute the finite sum at a vector :math:`z = x + iy`.

    Parameters
    ----------
    X : double[:,:]
        Real part of the Riemann matrix.
    Yinv : double[:,:]
        Inverse of the imaginary part of the Riemann matrix.
    T : double[:,:]
        Cholesky decomposotion of the imaginary part of the Riemann matrix.
    x : double[:]
        Real part of the input vector.
    y : double[:]
        Imaginary part of the input vector.
    S : int[:,:]
        Set of integer points over which to compute the finite sum.
    g : int
        Genus.
    N : int
        Number of integer points in `S`.
    """
    cdef double* real
    cdef double* imag
    cdef complex value
    finite_sum_without_derivatives(real, imag, &X[0,0], &Yinv[0,0], &T[0,0],
                                   &x[0], &y[0], &S[0,0], g, N)
    value = real[0] + 1.0j*imag[0]
    return value


cpdef complex finite_sum_derivatives(double[:,:] X, double[:,:] Yinv,
                                     double[:,:] T, double[:] x, double[:] y,
                                     int[:,:] S, double[:,:] derivs_real,
                                     double[:,:] derivs_imag, int g, int N):
    r"""Foo

    """
    cdef double* real
    cdef double* imag
    cdef complex value
    finite_sum_with_derivatives(real, imag, &X[0,0], &Yinv[0,0], &T[0,0],
                                   &x[0], &y[0], &S[0,0], &derivs_real[0,0],
                                   &derivs_imag[0,0], g, N)
    value = real[0] + 1.0j imag[0]
    return value


cpdef complex oscillatory_part(z, Omega, epsilon=1e-8, derivs=[],
                               accuracy_radius=5):
    cdef double* real     # real part of result
    cdef double* imag     # imag part of result
    cdef complex value    # result
    cdef double[:,:] X    # real part of Omega
    cdef double[:,:] Yinv # inv of imag part of Omega
    cdef double[:,:] T    # Cholesky decomp. of imag part of Omega
    cdef double[:,:] x    # real part of z
    cdef double[:,:] y    # imag part of z
    cdef int[:,:] S       # integer points
    cdef int g            # genus
    cdef int N            # number of integer points

    # Omega argument info
    Omega = numpy.array(Omega, dtype=numpy.complex)
    g = Omega.shape[0]
    Yinv = numpy.linalg.inv(Omega.imag)
    T = numpy.linalg.cholesky(Omega.imag)

    # z argument info
    z = numpy.array(z, dtype=numpy.complex)
    x = z.real
    y = z.imag

    # integer point info
    R = raidus(epsilon, T, derivs=derivs, accuracy_radius=accuracy_radius)
    S = _find_integer_points(g, R, T)
    N = S.size/g;

    if derivs == []:
        finite_sum_without_derivatives(real, imag, &X[0,0], &Yinv[0,0],
                                       &T[0,0], &x[0], &y[0], &S[0,0], g, N)
    else:
        derivs = numpy.array(derivs, dtype=numpy.complex)
        cdef double[:,:] derivs_real = derivs.real
        cdef double[:,:] derivs_imag = derivs.imag
        finite_sum_with_derivatives(real, imag, &X[0,0], &Yinv[0,0], &T[0,0],
                                    &x[0], &y[0], &S[0,0], &derivs_real[0,0],
                                    &derivs_imag[0,0], g, N)

    value = real[0] + 1.0j*imag[0]
    return value


cdef class RiemannTheta_Function(object):

    # cached properties:
    # * recompute integer points if __Omega changes
    @property
    def _Omega(self):
        return self.__Omega
    @_Omega.setter
    def _Omega(self, value):
        self.__Omega = value

    def __init__(self, accuracy_radius=5):
        pass

    def __call__(self, z, Omega, **kwds):
        u = self.exponential_part(z, Omega, **kwds)
        v = self.oscillatory_part(z, Omega, **kwds)
        return (u,v)

    def exponential_part(self, z, Omega, **kwds):
        pass

    def oscillatory_part(self, z, Omega, **kwds):
        pass




# declaration of Riemann theta
RiemannTheta = RiemannTheta_Function()
