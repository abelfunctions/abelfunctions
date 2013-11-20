cdef class MultivariatePolynomial:
    cdef object c
    cdef int deg
    cdef complex eval(self,complex z1,complex z2)

cdef int factorial(int n) nogil

cdef complex newton(object df,
                    complex xip1,
                    complex yij)

cdef double smale_beta(object df,
                       complex xip1,
                       complex yij)

cdef double smale_gamma(object df,
                        complex xip1,
                        complex yij)

cdef double smale_alpha(object df,
                        complex xip1,
                        complex yij)
