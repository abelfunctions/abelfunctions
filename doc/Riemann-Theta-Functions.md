Riemann Theta Functions
=======================


Computing Riemann Theta Functions
---------------------------------

To compute the Riemann theta function we first need to create a Riemann matrix.
`RiemannTheta` requires a Numpy matrix:

    >>> from abelfunctions import RiemannTheta
    >>> import numpy
    >>> Omega = numpy.matrix([[1j,-0.5],[-0.5,1j]], dtype=numpy.complex)
    >>> z = [0.1+0.2j, 0.3+0.4j]
    >>> RiemannTheta(z, Omega, prec=1e-16)
    (1.02975400265-0.532395718212j)