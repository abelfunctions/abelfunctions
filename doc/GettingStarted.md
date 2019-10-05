# Getting Started

This document presents a brief overview of the capabilities of Abelfunctions by working through a simple example.

## Riemann Surfaces

One of the primary objects we can construct with abelfunctions is a Riemann surface. In particular, a Riemann surface *X* obtained by desingularizing and compactifying a complex plane algebraic curve *f(x,y) = 0*. To create a Riemann surface begin with bivariate polynomial over the rational numbers.

```python
sage: from abelfunctions import *
sage: R.<x,y> = QQ[]
sage: f = y**3 + 2*x**3*y - x**7
sage: X = RiemannSurface(f)
sage: X
Riemann surface defined by f = -x^7 + 2*x^3*y + y^3
```

Abelfunctions can compute the singularities of the curve and uses the corresponding information to determine the genus.

```python
sage: X.genus()
2
```

## Holomorphic Differentials

We can compute a basis for the space of holomorphic differentials on a Riemann surface. The affine part of the differentials are displayed when computed.

```python
sage: differentials = X.differentials
sage: for omega in differentials:
....:     print(omega)
x*y/(2*x^3 + 3*y^2)
x^3/(2*x^3 + 3*y^2)
```

## Homology Basis

Dual to the holomorphic differentials is the first homology group of *X*. This space is spanned by a canonical basis of cycles *{a_1, ..., a_g, b_1, ..., b_g}*.

[IMAGE]

We can compute and plot the projections of these cycles in complex *x-* and *y-* planes.

```python
sage: a_cycles = X.a_cycles()
sage: b_cycles = X.b_cycles()
sage: xfig = a_cycles[0].plot_x(512)
sage: yfig = a_cycles[0].plot_y(512, color='green')
sage: xfig.show(); yfig.show()
```

[IMAGE]
[IMAGE]


## Period Matrices and Riemann Matrices

Using the above two ingredients, we can compute the period matrix. This is done by calling `RiemannSurface.period_matrix()`. Normalized *b-*periods are obtained by calling `RiemannSurface.riemann_matrix()`.

```python
sage: import numpy
sage: numpy.set_printoptions(precision=6) # truncate numerical output
sage: tau = X.period_matrix(); tau
[[ -1.864991e-11-1.201925j   1.849572e+00+0.600962j
   -7.064736e-01+2.174302j  -1.849572e+00+2.545717j]
 [ -1.885112e-11+1.971464j   7.161762e-01-0.985732j
   -1.874974e+00-1.362248j  -7.161762e-01+0.2327j  ]]
sage: Omega = X.riemann_matrix(); Omega
[[-1.309017+0.951057j -0.809017+0.587785j]
 [-0.809017+0.587785j -1.000000+1.175571j]]
```

We numerically verify that `Omega` is a Riemann matrix: a complex *g x g* symmetric matrix with positive definite imaginary part.

```python
sage: import numpy.linalg
sage: numpy.linalg.norm(Omega - Omega.T)
1.22571092827e-10
sage: numpy.linalg.eigvals(Omega.imag)
[ 0.464905  1.661722]
```

## Riemann Theta Functions

Another major feature of Abelfunctions is the ability to efficiently compute the Riemann theta function.

[IMAGE]

```python
z = [0.5, 0.5*1.0j]
RiemannTheta(z,Omega)
(1.1141548131726919+0.88244003476732269j)
```

Abelfunctions is very efficient in computing the Riemann theta function for many values of *z*. Here we plot the real and imaginary parts of the above Riemann theta function as the first component of *z* varies over the complex interval *[-1,1] + [0.1,0.6]I* and the second component fixed at zero.

First, we setup a grid of complex numbers.

```python
sage: N = 128
sage: x = numpy.linspace(-1,1,N)
sage: y = numpy.linspace(0.1,0.6,N)
sage: X,Y = numpy.meshgrid(x,y)
sage: Z = X + 1.0j*Y
sage: Z = Z.flatten()
```

Next, we evaluate the Riemann theta function.

```python
sage: U = RiemannTheta([[z,0] for z in Z], Omega)
sage: U = U.reshape(N,N)
```

Finally, we plot the real and imaginary parts of the Riemann theta function at
each of these values.

```python
sage: import matplotlib
sage: import matplotlib.pyplot as plt
sage: fig = plt.figure(figsize=(16,6))
sage: ax_real = fig.add_subplot(1,2,1, projection='3d')
sage: ax_real.plot_surface(X, Y, U.real, cmap='jet')
sage: ax_imag = fig.add_subplot(1,2,2, projection='3d')
sage: ax_imag.plot_surface(X, Y, U.imag, cmap='jet')
sage: fig.show()
```

[IMAGE]

## Derivatives of Riemann Theta Functions

Abelfunctions is also able to compute directional derivatives of Riemann theta functions.
To we provide the directional derivatives as a list of lists to the `derivs` argument of the `RiemannTheta` function.
For example, suppose the Riemann matrix, `Omega`, is given by
```python
sage: import numpy as np
sage: r3 = sqrt(-3)/3
sage: Omega = np.array([[1+2*r3, -1-r3],
....:                   [-1-r3, 1+2*r3]])
```
To evaluate the directional derivative [1.2,2].grad of the Riemann theta function at the place z=[0,1] we would use
```python
sage: RiemannTheta([0, 1], Omega, derivs=[[1.2,2]])
(9.7394378275290653e-17-1.1491767560254569e-17j)
```
To evaluate the second order directional derivative ([1,0].grad)([0,1].grad) at z=[0,1] we would use
```python
sage: RiemannTheta([0, 1], Omega, derivs=[[1,0],[0,1]])
(-2.0914115976630612-1.4526163881071635e-18j)
```

