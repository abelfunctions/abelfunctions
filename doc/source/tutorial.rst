Tutorial
========

Here we give a brief overview of the capabilities of `abelfunctions` by
working through a simple example.

Riemann Surfaces
----------------

One of the primary objects we can construct with `abelfunctions` is a
Riemann surface. In particular, a Riemann surface :math:`X` obtained by
desingularizing and compactifying a complex plane algebraic curve
:math:`f(x,y) = 0`. To create a Riemann surface, provide a SymPy
(http://sympy.org) expression.

.. code-block:: python

    import sympy
    from sympy.abc import x,y
    from abelfunctions import *

    f = y**3 + 2*x**3*y - x**7
    X = RiemannSurface(f,x,y)
    print C

*
    .. code-block:: none

       Riemann surface defined by the algebraic curve -x**7 + 2*x**3*y + y**3

`abelfunctions` can compute the singularities of the curve and uses the
corresponding information to determine the genus.

.. code-block:: python

    print X.genus()

*
    .. code-block:: none

       2

Holomorphic Differentials
-------------------------

We can compute a basis for the space of holomorphic differentials
:math:`\{\omega_1, \ldots, \omega_g\}` defined on :math:`X`. The affine
part of the differentials are displayed when computed.

.. code-block:: python

    differentials = X.holomorphic_differentials()

    for omega in differentials:
        print omega

*
    .. code-block:: none

        x*y/(2*x**3 + 3*y**2)
        x**3/(2*x**3 + 3*y**2)


Homology Basis
--------------

Dual to the holomorphic differentials is the first Homology group of
:math:`X`. This space is spanned by a basis of cycles :math:`\{ a_1,
\ldots, a_g, b_1, \ldots, b_g \}`. In the genus two case, the picture
looks like so:

.. figure:: img/rs.png
    :figwidth: 100%
    :scale: 60%
    :align: center
    :alt: A genus two Riemann surface with cycle basis.

We can compute and plot these cycles in complex :math:`x-y` space.

.. code-block:: python

    a_cycles = X.a_cycles()
    b_cycles = X.b_cycles()

    # plot the x-part and y-parts of the first
    # a-cycle using 512 interpolation points
    xfig = a_cycles[0].plot_x(512)
    yfig = a_cycles[0].plot_y(512, color='g')
    xfig.show()
    yfig.show()

.. figure:: img/acycle_x.png
    :figwidth: 100%

.. figure:: img/acycle_y.png
    :figwidth: 100%


Period Matrices and Riemann Matrices
------------------------------------

Using the above two ingredients, we can compute the period matrix
:math:`\tau = [A \; | \; B] \in \mathbb{C}^{g \times 2g}` and Riemann
matrix :math:`\Omega \in \mathbb{C}^{g \times g}` of :math:`X` where

.. math::
    A_{ij} = \int_{a_j} \omega_i, \quad B_{ij} = \int_{b_j} \omega_i,

    \Omega = A^{-1} B.

.. code-block:: python

    import numpy
    import numpy.linalg
    numpy.set_printoptions(precision=6)

    tau = X.period_matrix()
    print tau

*
    .. code-block:: none

        [[ -1.381589e-12-1.201925j   1.849572e+00+0.600962j
           -7.064736e-01+2.174302j  -1.849572e+00+2.545717j]
         [  9.228812e-12+1.971464j   7.161762e-01-0.985732j
           -1.874974e+00-1.362248j  -7.161762e-01+0.2327j  ]]

.. code-block:: python

    Omega = X.riemann_matrix()
    print Omega

*
    .. code-block:: none

        [[-1.309017+0.951057j -0.809017+0.587785j]
         [-0.809017+0.587785j -1.000000+1.175571j]]

We numerically verify that :math:`\Omega` is indeed a Riemann matrix: a
complex :math:`g \times g` which is symmetric and with positive definite
imaginary part.

.. code-block:: python

    print numpy.linalg.norm(Omega - Omega.T)
    print
    print numpy.linalg.eigvals(Omega.imag)

*
    .. code-block:: none

        3.64209384448e-11

        [ 0.464905  1.661722]



Riemann Theta Functions
-----------------------

Another major feature of `abelfunctions` is the ability to compute the
Riemann theta function :math:`\theta : \mathbb{C}^g \times
\mathfrak{h}_g`

.. math::

    \theta(z,\Omega) = \sum_{n \in \mathbb{Z}^g} e^{2\pi i \left(
                       \frac{1}{2} n \cdot \Omega n + n \cdot z \right) }

where :math:`\mathfrak{h}_g` is the space of :math:`g \times g` Riemann
matrices. Using the Riemann matrix computed above we can compute
:math:`\theta(z,\Omega)` for various :math:`z \in \mathbb{C}^2`.

.. code-block:: python

    z = [0.5,0.5*1.0j]
    print RiemannTheta(z,Omega)

*
    .. code-block:: none

        (9.12688266829e-12+9.12688266829e-12j)

`abelfunctions` is very efficient in computing the Riemann theta
function for many values of :math:`z`. Here we plot the real and
imaginary parts of :math:`\theta(z,\Omega)` for :math:`z = (x + iy, 0)`
with :math:`x \in [0,5], y \in [0,1]`.

.. code-block:: python

    import matplotlib
    import matplotlib.pyplot as plt

    # compute an N x N grid of complex numbers
    N = 128
    x = numpy.linspace(-1,1,N)
    y = numpy.linspace(0.1,0.6,N)
    X,Y = numpy.meshgrid(x,y)
    Z = X + 1.0j*Y
    Z = Z.flatten()

    # the "batch" flag enables efficient computation
    # for many different z-arguments
    U = RiemannTheta([[z,0] for z in Z], Omega, batch=True)
    U = U.reshape(N,N)

    # plot
    fig = plt.figure(figsize=(16,6))
    ax_real = fig.add_subplot(1,2,1, projection='3d')
    ax_real.plot_surface(X, Y, U.real, cmap='jet')

    ax_imag = fig.add_subplot(1,2,2, projection='3d')
    ax_imag.plot_surface(X, Y, U.imag, cmap='jet')

    fig.show()


.. figure:: img/riemanntheta_genus2.png
    :figwidth: 100%
    :width: 100%
