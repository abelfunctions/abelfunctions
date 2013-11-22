cdef class UnivariatePolynomial:
    cdef complex[:] c
    cdef int deg
    cdef complex eval(self,complex z) nogil

cdef class MultivariatePolynomial:
    cdef UnivariatePolynomial[:] c
    cdef int deg
    cdef complex eval(self,complex z1,complex z2)

cdef int factorial(int n) nogil

cdef complex newton(MultivariatePolynomial[:] df,
                    complex xip1,
                    complex yij)

cdef double smale_beta(MultivariatePolynomial[:] df,
                       complex xip1,
                       complex yij)

cdef double smale_gamma(MultivariatePolynomial[:] df,
                        complex xip1,
                        complex yij)

cdef double smale_alpha(MultivariatePolynomial[:] df,
                        complex xip1,
                        complex yij)
