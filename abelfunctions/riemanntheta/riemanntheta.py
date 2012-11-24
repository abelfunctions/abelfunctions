"""
Computing Riemann Theta Functions

This module implements the algorithms for computing Riemann theta
functions and their derivatives featured in the paper *"Computing
Riemann Theta Functions"* by Deconinck, Heil, Bobenko, van Hoeij, and
Schmies [CRTF].


**DEFINITION OF THE RIEMANN THETA FUNCTION:**


Let `g` be a positive integer, the *genus* of the Riemann theta
function.  Let `H_g` denote the Siegel upper half space of dimension
`g(g+1)/2` over `\CC` , that is the space of symmetric complex
matrices whose imaginary parts are positive definite.  When `g = 1`,
this is just the complex upper half plane.

The Riemann theta function `\theta : \CC^g \times H_g \to \CC` is
defined by the infinite series

.. math::

    \theta( z | \Omega ) = \sum_{ n \in \ZZ^g } e^{ 2 \pi i \left( \tfrac{1}{2} n \cdot \Omega n + n \cdot z \right) }

It is holomorphic in both `z` and `\Omega`. It is quasiperiodic in `z`
with respect to the lattice `\{ M + \Omega N | M,N \in \ZZ^g \}`,
meaning that `\theta(z|\Omega)` is periodic upon translation of `z` by
vectors in `\ZZ^g` and periodic up to a multiplicative exponential
factor upon translation of `z` by vectors in `\Omega \ZZ^g`. As a
consequence, `\theta(z | \Omega)` has exponential growth in the
imaginary parts of `z`.

When `g=1`, the Riemann theta function is the third Jacobi theta
function.

.. math::

    \theta( z | \Omega) = \theta_3(\pi z | \Omega) = 1 + 2 \sum_{n=1}^\infty e^{i \pi \Omega n^2} \cos(2 \pi n z)

Riemann theta functions are the fundamental building blocks for
Abelian functions, which generalize the classical elliptic functions
to multiple variables. Like elliptic functions, Abelian functions and
consequently Riemann theta functions arise in many applications such
as integrable partial differential equations, algebraic geometry, and
optimization.

For more information about the basic facts of and definitions
associated with Riemann theta funtions, see the Digital Library of
Mathematics Functions ``http://dlmf.nist.gov/21``.


**ALGORITHM:**


The algorithm in [CRTF] is based on the observation that the
exponential growth of `\theta` can be factored out of the sum. Thus,
we only need to find an approximation for the oscillatory part. The
derivation is omitted here but the key observation is to write `z = x
+ i y` and `\Omega = X + i Y` where `x`, `y`, `X`, and `Y` are real
vectors and matrices.  With the exponential growth part factored out
of the sum, the goal is to find the integral points `n \in \ZZ^g` such
that the sum over these points is within `O(\epsilon)` accuracy of the
infinite sum, for a given `z \in \CC^g` and numerical accuracy
`\epsilon`.

By default we use the uniform approximation formulas which use the
same integral points for all `z` for a fixed `\Omega`. This can be
changed by setting ``uniform=False``. This is ill-advised if you need
to compute the Riemann theta function for a fixed `\Omega` for many
different `z`.


**REFERENCES:**


- [CRTF] Computing Riemann Theta Functions. Bernard Deconinck, Matthias 
  Heil, Alexander Bobenko, Mark van Hoeij and Markus Schmies.  Mathematics
  of Computation 73 (2004) 1417-1442.  The paper is available at
  http://www.amath.washington.edu/~bernard/papers/pdfs/computingtheta.pdf. 
  Accompanying Maple code is available at 
  http://www.math.fsu.edu/~hoeij/RiemannTheta/

- Digital Library of Mathematics Functions - Riemann Theta Functions ( http://dlmf.nist.gov/21 ).
 


**AUTHORS:**


- Chris Swierczewski (2011-11): major overhaul to match notation of
  [CRTF], numerous bug fixes, documentation, doctests, symbolic
  evaluation

 
"""
import numpy as np
import scipy as sp
import scipy.linalg as la
import RIEMANN
from scipy.special import gamma, gammaincc, gammainccinv
from scipy.optimize import fsolve
from riemanntheta_misc import *
import time


class RiemannTheta_Function:
    r"""
    Creates an instance of the Riemann theta function parameterized by a
    Riemann matrix ``Omega``, directional derivative ``derivs``, and
    derivative evaluation accuracy radius. See module level documentation
    for more information about the Riemann theta function.

    The Riemann theta function `\theta : \CC^g \times H_g \to \CC` is defined 
    by the infinite series

    .. math::

        \theta( z | \Omega ) = \sum_{ n \in \ZZ^g } e^{ 2 \pi i \left( \tfrac{1}{2} \langle \Omega n, n \rangle +  \langle z, n \rangle \right) }


    The precision of Riemann theta function evaluation is determined by
    the precision of the base ring.

    As shown in [CRTF], `n` th order derivatives introduce polynomial growth in
    the oscillatory part of the Riemann theta approximations thus making a
    global approximation formula impossible. Therefore, one must specify
    a ``deriv_accuracy_radius`` of guaranteed accuracy when computing 
    derivatives of `\theta(z | \Omega)`.

    INPUT:

    - ``Omega`` -- a Riemann matrix (symmetric with positive definite imaginary part)

    - ``deriv`` -- (default: ``[]``) a list of `g`-tuples representing a directional derivative of `\theta`. A list of `n` lists represents an `n`th order derivative.
    
    - ``uniform`` -- (default: ``True``) a unform approximation allows the accurate computation of the Riemann theta function without having to recompute the integer points over which to take the finite sum. See [CRTF] for a more in-depth definition.

    - ``deriv_accuracy_radius`` -- (default: 5) the guaranteed radius of accuracy in computing derivatives of theta. This parameter is necessary due to the polynomial growth of the non-doubly exponential part of theta



    OUTPUT:


    - ``Function_RiemannTheta`` -- a Riemann theta function parameterized by the Riemann matrix `\Omega`, derivatives ``deriv``, whether or not to use a uniform approximation, and derivative accuracy radius ``deriv_accuracy_radius``.


    .. note::

        For now, only second order derivatives are implemented. Approximation
        formulas are derived in [CRTF]. It is not exactly clear how to
        generalize these formulas. In most applications, second order
        derivatives are suficient.

    """

    def __init__(self, uniform=True, deriv_accuracy_radius=5):
        """
        Defines parameters in constructed class instance.
        """
        self.uniform               = uniform
        self.deriv_accuracy_radius = deriv_accuracy_radius

        # cache radii, intpoints, and inverses
        self._rad       = None
        self._intpoints = None
        self._Omega     = None
        self._Yinv      = None
        self._T         = None
        self._Tinv      = None


    def lattice(self):
        r"""
        Compute the complex lattice corresponding to the Riemann matix.

        .. note::

            Not yet implemented.
        """
        raise NotImplementedError()


    def genus(self):
        r"""
        The genus of the algebraic curve from which the Riemann matrix is
        calculated. If $\Omega$ is not block decomposable then this is just
        the dimension of the matrix.

        .. note::

            Block decomposablility detection is difficult and not yet 
            implemented. Currently, ``self.genus()`` just returns the size 
            of the matrix.            
        """
        return NotImplementedError()


    def integer_points(self, Yinv, T, Tinv, z, g, R):
        r"""
        The set, `U_R`, of the integral points needed to compute Riemann 
        theta at the complex point $z$ to the numerical precision given
        by the Riemann matirix base field precision.

        The set `U_R` of [CRTF], (21).

        .. math::
        
            \left\{ n \in \ZZ^g : \pi ( n - c )^{t} \cdot Y \cdot 
            (n - c ) < R^2, |c_j| < 1/2, j=1,\ldots,g \right\}

        Since `Y` is positive definite it has Cholesky decomposition 
        `Y = T^t T`. Letting `\Lambda` be the lattice of vectors 
        `v(n), n \in ZZ^g` of the form `v(n)=\sqrt{\pi} T (n + [[ Y^{-1} n]])`,
        we have that 

        .. math::

            S_R = \left\{ v(n) \in \Lambda : || v(n) || < R \right\} .

        Note that since the integer points are only required for oscillatory
        part of Riemann theta all over these points are near the point 
        `0 \in \CC^g`. Additionally, if ``uniform == True`` then the set of
        integer points is independent of the input points `z \in \CC^g`.

        .. note::
        
            To actually compute `U_R` one needs to compute the convex hull of
            `2^{g}` bounding ellipsoids. Since this is computationally
            expensive, an ellipsoid centered at `0 \in \CC^g` with large
            radius is computed instead. This can cause accuracy issues with
            ill-conditioned Riemann matrices, that is, those that produce
            long and narrow bounding ellipsoies. See [CRTF] Section ### for
            more information.

        INPUTS:

        - ``Yinv`` -- the inverse of the imaginary part of the Riemann matrix
          `\Omega`

        - ``T`` -- the Cholesky decomposition of the imaginary part of the
          Riemann matrix `\Omega`

        - ``z`` -- the point `z \in \CC` at which to compute `\theta(z|\Omega)`
         
        - ``R`` -- the first ellipsoid semi-axis length as computed by ``self.radius()``
        """
        g    = Yinv.shape[0]
        pi   = np.pi
        z    = np.array(z).reshape((g,1))
        x    = z.real
        y    = z.imag
        
        # determine center of ellipsoid.
        if self.uniform:
            c     = np.zeros((g,1))
            intc  = np.zeros((g,1))
            leftc = np.zeros((g,1))
        else:
            c     = Yinv * y
            intc  = c.round()
            leftc = c - intc


        def find_integer_points(g, c, R, start):
            r"""
            Recursion function for computing the integer points needed in 
            each coordinate direction.

            INPUT:
            
            - ``g`` -- the genus. recursively used to determine integer 
              points along each axis.

            - ``c`` -- center of integer point computation. `0 \in \CC^g` 
              is used when using the uniform approximation.

            - ``R`` -- the radius of the ellipsoid along the current axis.

            - ``start`` -- the starting integer point for each recursion 
              along each axis.

            OUTPUT:

            - ``intpoints`` -- (list) a list of all of the integer points 
              inside the bounding ellipsoid along a single axis

            ... todo::

                Recursion can be memory intensive in Python. For genus `g<30`
                this is a reasonable computation but can be sped up by 
                writing a loop instead.
            """
            a = int( np.ceil((c[g] - R/T[g,g]).real)  )
            b = int( np.floor((c[g] + R/T[g,g]).real) )

            # check if we reached the edge of the ellipsoid
            if not a < b: return np.array([])
            # last dimension reached: append points
            if g == 0:
                return np.array([np.append([i],start) for i in xrange(a,b+1)])
        
            #
            # compute new shifts, radii, start, and recurse
            #
            newg    = g-1
            newT    = T[:(newg+1),:(newg+1)]
            newTinv = la.inv(newT)
            pts     = []

            for n in xrange(a, b+1):
                chat     = c[:newg+1]
                that     = T[:newg+1,g]
                newc     = chat - newTinv * that * (n - c[g])
                newR     = np.sqrt(R**2 - (T[g,g] * (n - c[g]))**2)  # XXX
                newstart = np.append([n],start)
                newpts   = find_integer_points(newg,newc,newR,newstart)
                pts      = np.append(pts,newpts)

            return pts

        return find_integer_points(g-1, leftc, R, np.array([]))

    
    def radius(self, T, prec, deriv=[]):
        r"""
        Calculate the radius `R` to compute the value of the theta function
        to within `2^{-P + 1}` bits of precision where `P` is the 
        real / complex precision given by the input matrix. Used primarily
        by ``RiemannTheta.integer_points()``.

        `R` is the radius of [CRTF] Theorems 2, 4, and 6.

        Input
        -----
        
        - ``T`` -- the Cholesky decomposition of the imaginary part of the 
          Riemann matrix `\Omega`

        - ``prec`` -- the desired precision of the computation
        
        - ``deriv`` -- (list) (default=``[]``) the derivative, if given. 
          Radius increases as order of derivative increases.            
        """
        Pi = np.pi
        I  = 1.0j
        g  = np.float64(T.shape[0])

        # compute the length of the shortest lattice vector
        #U  = qflll(T)
	U = T
        A  = U*T
        r  = min(la.norm(A[:,i]) for i in range(int(g)))
        normTinv = la.norm(la.inv(T))

        # solve for the radius using:
        #   * Theorem 3 of [CRTF] (no derivative)
        #   * Theorem 5 of [CRTF] (first order derivative)
        #   * Theorem 7 of [CRTF] (second order derivative)
        if len(deriv) == 0:
            eps  = prec
            lhs  = eps * (2.0/g) * (r/2.0)**g * gamma(g/2.0)
            ins  = gammainccinv(g/2.0,lhs)
            R    = np.sqrt(ins) + r/2.0
            rad  = max( R, (np.sqrt(2*g)+r)/2.0 )
        elif len(deriv) == 1:
            # solve for left-hand side
            L         = self.deriv_accuracy_radius
            normderiv = la.norm(np.array(deriv[0]))
            eps  = prec
            lhs  = (eps * (r/2.0)**g) / (np.sqrt(Pi)*g*normderiv*normTinv)

            # define right-hand-side function involving the incomplete gamma
            # function
            def rhs(ins):
                """
                Right-hand side function for computing the bounding ellipsoid
                radius given a desired maximum error bound for the first
                derivative of the Riemann theta function.
                """
                return gamma((g+1)/2)*gammaincc((g+1)/2, ins) +               \
                    np.sqrt(Pi)*normTinv*L * gamma(g/2)*gammaincc(g/2, ins) - \
                    float(lhs)

            #  define lower bound (guess) and attempt to solve for the radius
            lbnd = np.sqrt(g+2 + np.sqrt(g**2+8)) + r
            try:
                ins = fsolve(rhs, float(lbnd))[0]
            except RuntimeWarning:
                # fsolve had trouble finding the solution. We try 
                # a larger initial guess since the radius increases
                # as desired precision increases
                try:
                    ins = fsolve(rhs, float(2*lbnd))[0]
                except RuntimeWarning:
                    raise ValueError, "Could not find an accurate bound for the radius. Consider using higher precision."

            # solve for radius
            R   = np.sqrt(ins) + r/2.0
            rad = max(R,lbnd)

        elif len(deriv) == 2:
            # solve for left-hand side
            L             = self.deriv_accuracy_radius
            prodnormderiv = prod([la.norm(d) for d in deriv])

            eps  = prec
            lhs  = (eps*(r/2.0)**g) / (2*Pi*g*prodnormderiv*normTinv**2)

            # define right-hand-side function involving the incomplete gamma
            # function
            def rhs(ins):
                """
                Right-hand side function for computing the bounding ellipsoid
                radius given a desired maximum error bound for the second
                derivative of the Riemann theta function.
                """
                return gamma((g+2)/2)*gammaincc((g+2)/2, ins) + \
                    2*np.sqrt(Pi)*normTinv*L *                  \
                    gamma((g+1)/2)*gammaincc((g+1)/2,ins) +     \
                    Pi*normTinv**2*L**2 *                       \
                    gamma(g/2)*gammaincc(g/2,ins) - float(lhs)

            #  define lower bound (guess) and attempt to solve for the radius
            lbnd = np.sqrt(g+4 + np.sqrt(g**2+16)) + r
            try:
                ins = fsolve(rhs, float(lbnd))[0]
            except RuntimeWarning:
                # fsolve had trouble finding the solution. We try 
                # a larger initial guess since the radius increases
                # as desired precision increases
                try:
                    ins = fsolve(rhs, float(2*lbnd))[0]
                except RuntimeWarning:
                    raise ValueError, "Could not find an accurate bound for the radius. Consider using higher precision."

            # solve for radius
            R   = np.sqrt(ins) + r/2.0
            rad = max(R,lbnd)

        else:
            # can't computer higher derivatives, yet
            raise NotImplementedError("Ellipsoid radius for first and second derivatives not yet implemented.")

        return rad


    def exp_and_osc_at_point(self, z, Omega, prec=1e-8, deriv=[], gpu=False):
        r"""
        Calculate the exponential and oscillating parts of `\theta(z,\Omega)`.
        (Or a given directional derivative of `\theta`.) That is, compute 
        complex numbers `u,v \in \CC` such that `\theta(z,\Omega) = e^u v` 
        where the value of `v` is oscillatory as a function of `z`.            
        """
        g = Omega.shape[0]
        pi = np.pi

        # perform some simple cacheing on the matrices
        X = np.matrix(Omega.real)
        Y = np.matrix(Omega.imag)
        Yinv = np.matrix(la.inv(Y))
        T = np.matrix(la.cholesky(Y))
        Tinv = np.matrix(la.inv(T))
            
        # extract real and imaginary parts of input z
        z = np.array(z).reshape((g,1))
        x = z.real
        y = z.imag

        # convert derivatives to vector type       
        deriv = np.array(deriv)

        # compute integer points: check for uniform approximation
        if self.uniform:
            # check if we've already computed the uniform radius and intpoints
            if self._rad is None:
                self._rad = self.radius(T, prec, deriv=deriv)
            if self._intpoints is None:
                origin          = [0]*g
                start = time.clock()
                self._intpoints = self.integer_points(Yinv, T, Tinv, origin, 
                                                      g, self._rad)
                print (time.clock() - start)
            R = self._rad
            S = self._intpoints
        else:
            R = self.radius(T, prec, deriv=deriv)
            S = self.integer_points(Yinv, T, Tinv, z, g, R)

        # compute oscillatory and exponential terms
        if gpu:
            from riemanntheta_misc import finite_sum_opencl
            v = finite_sum_opencl(X, Yinv, T, x, y, S, g)
        elif (len(deriv) > 0):
            v = RIEMANN.finite_sum_derivatives(X, Yinv, T, x, y, S, deriv, g)
        else:
            v = RIEMANN.finite_sum(X, Yinv, T, x, y, S, g)
        u = pi*np.dot(y.T,Yinv * y).item(0,0)

        return u,v

    def value_at_point(self, z, Omega, prec=1e-8, deriv=[], gpu=False):
        r"""
        Returns the value of `\theta(z,\Omega)` at a point `z`.
        """

        exp_part, osc_part = self.exp_and_osc_at_point(z, Omega, prec=prec,
                                                       deriv=deriv, gpu=gpu)
        return np.exp(exp_part) * osc_part

    def __call__(self, z, Omega, prec=1e-8, deriv=[], gpu=False):
        r"""
        Returns the value of `\theta(z,\Omega)` at a point `z`. Lazy evaluation
        is done if the input contains symbolic variables.            
        """
        return self.value_at_point(z, Omega, prec=prec, deriv=deriv, gpu=gpu)


# declaration of Riemann theta
RiemannTheta = RiemannTheta_Function()
        

if __name__=="__main__": 
    print "=== Riemann Theta ==="
    theta = RiemannTheta
    z = np.array([0,0])
    Omega = np.matrix([[1.0j,-0.5],[-0.5,1.0j]])

    print "Test #1:"
    print theta.value_at_point(z,Omega,gpu=False)
    print "1.1654 - 1.9522e-15*I"
    print 

    print "Test #2:"
    z = np.array([1.0j,1.0j])
    u,v = theta.exp_and_osc_at_point(z,Omega,gpu=False)
    print theta.value_at_point(z,Omega,gpu=False)
    print "-438.94 + 0.00056160*I"
    
    print
    print "Derivative Tests:"
    print "Calculating directional derivatives at z = [i, 0]"
    print
    y = np.array([1.0j, 0])
    print "For [[1,0]]:"
    print theta.value_at_point(y, Omega, deriv = [[1,0]], gpu = False)
    print "0 - 146.49i"
    print
    print "For [[1,0] , [1,0]]: "
    print theta.value_at_point(y, Omega, deriv = [[1,0], [0,1]], gpu = False)
    print "0 + 0i" 
    print
    print "For [[0,1], [1,0]]: "
    print theta.value_at_point(y, Omega, deriv = [[0,1], [1,0]], gpu = False)
    print "0 + 0i"
    print
    print "For [[1,0],[1,0],[1,1]]:"
    print theta.value_at_point(y, Omega, deriv = [[1,0], [1,0], [1,1]], gpu = False)
    print "0 + 7400.39i" 
    print
    print "For [[1,1],[1,1],[1,1],[1,1]]: "
    print theta.value_at_point(y, Omega, deriv = [[1,1],[1,1],[1,1],[1,1]], gpu = False)
    print "41743.92 + 0i" 
    print
   
    print "Test #3"
    import pylab as p
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt

    print "\tCalculating theta..."
    f = lambda x,y: theta.exp_and_osc_at_point([x+1.0j*y,0],Omega,gpu=False)[1]
    f = np.vectorize(f)
    x = np.linspace(0,1,60)
    y = np.linspace(0,5,60)
    X,Y = p.meshgrid(x,y)
    Z = np.real(f(X,Y))

    print "\tPlotting..."
    plt.contourf(X,Y,Z,7,antialiased=True)
    plt.show()
 


                       

