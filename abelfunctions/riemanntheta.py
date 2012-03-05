r"""
Computing Riemann Theta Functions

This module implements the algorithms for computing Riemann theta functions
and their derivatives featured in the paper *"Computing Riemann Theta 
Functions"* by Deconinck, Heil, Bobenko, van Hoeij, and Schmies [CRTF].


**DEFINITION OF THE RIEMANN THETA FUNCTION:**


Let `g` be a positive integer, the *genus* of the Riemann theta
function.  Let `H_g` denote the Siegel upper half space of dimension 
`g(g+1)/2` over `\CC` , that is the space of symmetric complex matrices whose
imaginary parts are positive definite.  When `g = 1`, this is just the complex 
upper half plane.

The Riemann theta function `\theta : \CC^g \times H_g \to \CC`
is defined by the infinite series

.. math::

    \theta( z | \Omega ) = \sum_{ n \in \ZZ^g } e^{ 2 \pi i \left( \tfrac{1}{2} n \cdot \Omega n + n \cdot z \right) }

It is holomorphic in both `z` and `\Omega`. It is quasiperiodic in `z` with
respect to the lattice `\{ M + \Omega N | M,N \in \ZZ^g \}`, meaning that
`\theta(z|\Omega)` is periodic upon translation of `z` by vectors in `\ZZ^g`
and periodic up to a multiplicative exponential factor upon translation of `z`
by vectors in `\Omega \ZZ^g`. As a consequence, `\theta(z | \Omega)` has
exponential growth in the imaginary parts of `z`.

When `g=1`, the Riemann theta function is the third Jacobi theta function.

.. math::

    \theta( z | \Omega) = \theta_3(\pi z | \Omega) = 1 + 2 \sum_{n=1}^\infty e^{i \pi \Omega n^2} \cos(2 \pi n z)

Riemann theta functions are the fundamental building blocks for Abelian 
functions, which generalize the classical elliptic functions to multiple 
variables. Like elliptic functions, Abelian functions and consequently 
Riemann theta functions arise in many applications such as integrable
partial differential equations, algebraic geometry, and optimization.

For more information about the basic facts of and definitions associated with
Riemann theta funtions, see the Digital Library of Mathematics Functions
``http://dlmf.nist.gov/21``.


**ALGORITHM:**


The algorithm in [CRTF] is based on the observation that the exponential
growth of `\theta` can be factored out of the sum. Thus, we only need to
find an approximation for the oscillatory part. The derivation is 
omitted here but the key observation is to write `z = x + i y` and
`\Omega = X + i Y` where `x`, `y`, `X`, and `Y` are real vectors and matrices.
With the exponential growth part factored out of the sum, the goal is to find 
the integral points `n \in \ZZ^g` such that the sum over these points is
within `O(\epsilon)` accuracy of the infinite sum, for a given `z \in \CC^g`
and numerical accuracy `\epsilon`.

By default we use the uniform approximation formulas which use the same integral points for all `z` for a fixed `\Omega`. This can be 
changed by setting ``uniform=False``. This is ill-advised if you need
to compute the Riemann theta function for a fixed `\Omega` for many different `z`.


**EXAMPLES:**


We start by creating a genus 2 Riemann theta function from a Riemann matrix::

    sage: R = ComplexField(20); I = R.gen()
    sage: Omega = matrix(R,2,2,[I,-1/2,-1/2,I])
    sage: theta = RiemannTheta(Omega)
    sage: theta
    Riemann theta function with defining Riemann matrix
    [1.0000*I -0.50000]
    [-0.50000 1.0000*I]
    over the base ring Complex Field with 20 bits of precision

Since ``Omega`` above is defined over the complex field with 20 bits of 
precision, ``RiemannTheta`` can be evaluated at any point in `\CC^2` with 
the same precision. (These values are checked against the Maple
implementation of Riemann theta written by Bernard Deconinck and Mark van 
Hoeij; two of the authors of [CRTF].)::

    sage: theta([0,0])
    1.1654 - 1.9522e-15*I
    sage: theta([I,I])
    -438.94 + 0.00056160*I

One can also compute the exponential and oscillatory parts of the Riemann
theta function separately::

    sage: u,v = theta.exp_and_osc_at_point([I,I])
    sage: (u,v)
    (6.2832, -0.81969 + 1.0488e-6*I)
    sage: e^u*v
    -438.94 + 0.00056160*I

Directional derivatives of theta can also be computed. The directional 
derivative can be specified in the construction of the Riemann theta function
or as input to evaluation::

    sage: theta10 = RiemannTheta(Omega, deriv=[[1,0]])
    sage: z = [I,I]
    sage: theta10(z)
    0.0031244 + 2757.9*I
    sage: theta = RiemannTheta(Omega)
    sage: theta(z,[[1,0]])
    0.0031244 + 2757.9*I

Symbolic evaluation and differentiation work::

    sage: f = theta([x^2,pi*sin(x)]); f
    theta(x^2, pi*sin(x))
    sage: fx = f.derivative(x,1); fx
    pi*cos(x)*theta_01(x^2, pi*sin(x)) + 2*x*theta_10(x^2, pi*sin(x))
    sage: w = fx(x=I); w
    -(1.0631e7 + 1.4805e20*I)*pi - 6.2625e11 - 7.0256e12*I
    sage: CC(w)
    -6.26279686008491e11 - 4.65101957990761e20*I

It is important to note that the absolute value of the "oscillatory" part 
of the Riemann theta function grows polynomially with degree equal to the
number of derivatives taken. (e.g. the absolute value of the oscillatory part
of the first directional derivative of the function grows linearly) Therefore,
a radius of accuracy (Default: 5) must be specified to ensure that the value
of the derivative(s) of the Riemann theta function for `z` in a sphere of this
radius in `\ZZ^g` are accurate to within the desired numerical accuracy
specified by the base field of the Riemann matrix.

This radius of accuracy for values of the derivatives of the Riemann theta
function can be adjusted::

    sage: theta01 = RiemannTheta(Omega, deriv=[[0,1]], deriv_accuracy_radius=2)
    sage: theta01([0.3,0.4*I])   # guaranteed accurate to 20 bits
    2.6608e-8 - 3.4254*I
    sage: z = [1.7+2.3*I,3.9+1.7*I]
    sage: theta01(z)             # not guaranteed accurate to 20 bits
    -9.5887e11 - 4.7112e11*I

TESTS:

    sage: loads(dumps(RiemannTheta)) == RiemannTheta
    True
    

**REFERENCES:**


- [CRTF] Computing Riemann Theta Functions. Bernard Deconinck, Matthias 
  Heil, Alexander Bobenko, Mark van Hoeij and Markus Schmies.  Mathematics
  of Computation 73 (2004) 1417-1442.  The paper is available at
  http://www.amath.washington.edu/~bernard/papers/pdfs/computingtheta.pdf. 
  Accompanying Maple code is available at 
  http://www.math.fsu.edu/~hoeij/RiemannTheta/

- http://en.wikipedia.org/wiki/Theta_function

- http://mathworld.wolfram.com/JacobiThetaFunctions.html

- Digital Library of Mathematics Functions - Riemann Theta Functions ( http://dlmf.nist.gov/21 ).
 


**AUTHORS:**


- Nick Alexander (2009-03): initial version

- Chris Swierczewski (2011-11): major overhaul to match notation of [CRTF], numerous bug fixes, documentation, doctests, symbolic evaluation

 
"""

#*****************************************************************************
#       Copyright (C) 2012 Chris Swierczewski <cswiercz@gmail.com>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#                  http://www.gnu.org/licenses/
#*****************************************************************************

import pdb

import numpy as np
import scipy as sp
import scipy.linalg as la

#from abelfunctions.utilities import qflll
from utilities import qflll
from scipy.special import gamma, gammaincc, gammainccinv
from scipy.optimize import fsolve
from riemanntheta_misc import finite_sum

"""
from sage.calculus.var            import SR, var
from sage.ext.fast_callable       import fast_callable 
from sage.matrix.constructor      import matrix, identity_matrix
from sage.modules.free_module_element import vector
from sage.misc.misc_c             import prod
from sage.plot.all                import implicit_plot
from sage.rings.all               import CDF,RDF,RealField,ZZ,PolynomialRing
from sage.symbolic.function       import BuiltinFunction
from sage.symbolic.expression     import is_Expression
from scipy.optimize               import fsolve
from scipy.special                import gamma, gammaincc, gammainccinv
"""

class RiemannTheta:
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


    EXAMPLES:

    We start by creating a genus 2 Riemann theta function from a Riemann matrix.::

        sage: R = ComplexField(20); I = R.gen()
        sage: Omega = matrix(R,2,2,[I,-1/2,-1/2,I])
        sage: theta = RiemannTheta(Omega)
        sage: theta
        Riemann theta function with defining Riemann matrix
        [1.0000*I -0.50000]
        [-0.50000 1.0000*I]
        over the base ring Complex Field with 20 bits of precision

    Since ``Omega`` above is defined over the complex field with 20 bits of 
    precision, ``RiemannTheta`` can be evaluated at any point in `\CC^2` with 
    the same precision. (These values are checked against the Maple
    implementation of Riemann theta written by Bernard Deconinck and Mark van 
    Hoeij; two of the authors of [CRTF].)::
    
        sage: theta([0,0])
        1.1654 - 1.9522e-15*I
        sage: theta([I,I])
        -438.94 + 0.00056160*I

    One can also compute the exponential and oscillatory parts of the Riemann
    theta function separately::

        sage: u,v = theta.exp_and_osc_at_point([I,I])
        sage: (u,v)
        (6.2832, -0.81969 + 1.0488e-6*I)
        sage: e^u*v
        -438.94 + 0.00056160*I
        
    Directional derivatives of theta can also be computed. The directional 
    derivative can be specified in the construction of the Riemann theta 
    function or as input to evaluation.::
    
        sage: theta10 = RiemannTheta(Omega, deriv=[[1,0]])
        sage: z = [I,I]
        sage: theta10(z)
        0.0031244 + 2757.9*I
        sage: theta = RiemannTheta(Omega)
        sage: theta(z,[[1,0]])
        0.0031244 + 2757.9*I

    Symbolic evaluation and differentiation works::

        sage: f = theta([x^2,pi*sin(x)]); f
        theta(x^2, pi*sin(x))
        sage: fx = f.derivative(x,1); fx
        pi*cos(x)*theta_01(x^2, pi*sin(x)) + 2*x*theta_10(x^2, pi*sin(x))
        sage: w = fx(x=I); w
        -(1.0631e7 + 1.4805e20*I)*pi - 6.2625e11 - 7.0256e12*I
        sage: CC(w)
        -6.26279686008491e11 - 4.65101957990761e20*I

    It it important to note that the "oscillatory" part 
    of Riemann theta grows polynomially in absolute value where the degree of 
    the polynomial is equal to the degree of the directional derivative. (e.g. 
    the oscillatory part of the first directional derivative of theta grows 
    linearly in absolute value.) A radius of accuracy must therefore be 
    specified to ensure that the value of the derivative(s) of the Riemann 
    theta function are accurate. (Default: 5) This is the radius of the complex
    `g`-sphere where accuracy of the directional derivative of Riemann theta is
    guaranteed to be within the desired numerical accuracy specified by the 
    base field of the Riemann matrix.
    
    This radius of accuracy for the derivatives of the Riemann theta function 
    can be adjusted::

        sage: theta01 = RiemannTheta(Omega, deriv=[[0,1]], deriv_accuracy_radius=2)
        sage: theta01([0.3,0.4*I])   # guaranteed accurate to 20 bits
        2.6608e-8 - 3.4254*I
        sage: z = [1.7+2.3*I,3.9+1.7*I]
        sage: theta10(z)             # not guaranteed accurate to 20 bits
        -1.0103e12 - 4.6132e11*I

    TESTS:

        sage: sage.functions.riemann_theta.RiemannTheta_Constructor == RiemannTheta
        True
    """

    def __init__(self, uniform=True, deriv_accuracy_radius=5):
        """
        Defines parameters in constructed class instance.

        EXAMPLE:

        An example doesn't really make sense, so we demonstrate the 
        construction of a Riemann theta function::
        
            sage: R = ComplexField(20); I = R.gen()
            sage: Omega = matrix(R,2,2,[I,-1/2,-1/2,I])
            sage: theta = RiemannTheta(Omega, deriv=[[1,0]], deriv_accuracy_radius=6.0)
            sage: theta.deriv
            [[1, 0]]
            sage: theta.deriv_accuracy_radius
            6.00000000000000
                
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
        Compute the complex lattice corresponding to the Riemann matix. Uses 
        the :module:sage.functions.complex_lattice module.

        .. note::

            Not yet implemented.

        EXAMPLES::
        
            sage: from sage.functions.riemann_theta import RiemannTheta
            sage: R = ComplexField(20); I = R.gen()
            sage: Omega = matrix(R,2,2,[I,-1/2,-1/2,I])
            sage: theta = RiemannTheta(Omega)
            sage: theta.lattice() # optional: not implemented
            Traceback (most recent call last)
            ...
            NotImplementedError
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
            
        EXAMPLES:
            
        The following Riemann matrix is `2 \times 2` and is not block 
        decomposable. So its genus should be two::

            sage: from sage.functions.riemann_theta import RiemannTheta
            sage: R = ComplexField(36); I = R.gen()
            sage: Omega = matrix(R,2,2,[1.690983006 + 0.9510565162*I, 1.5 + 0.363271264*I, 1.5 + 0.363271264*I, 1.309016994+ 0.9510565162*I])
            sage: theta = RiemannTheta(Omega)
            sage: theta.genus()
            2
            
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


        OUTPUTS:

        - (list) -- a list of integer points in `\ZZ^g` that fall within the pointwise approximation ellipsoid defined in [CRTF]


        .. warning::

            At times we work over ``RDF``, which have very low 
            precision (53 bits). This could be a problem when given 
            ill-conditioned input. The general computing theta functions with
            such ill-conditioned input will not be possible, so
            we do not concern outselves with this case. This can be resolved
            by implementing the Siegel transformation discussed in [CRTF].


        EXAMPLES:
        ``integer_points()`` returns the points over which the finite sum
        is computed given the first major axis of the bounding ellipsoid.
        Here, we simply provide such a radius for testing purposes.::
        
            sage: from sage.functions.riemann_theta import RiemannTheta
            sage: R = ComplexField(36); I = R.gen()
            sage: Omega = matrix(R,2,2,[I,-1/2,-1/2,I])
            sage: theta = RiemannTheta(Omega)
            sage: theta.integer_points([0,0],2)
            [[-1, 0], [0, 0], [1, 0]]
            sage: theta.integer_points([0,0],3)
            [[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]

        """
        g    = Yinv.shape[0]
        pi   = np.pi
        z    = np.array(z)
        x    = z.real
        y    = z.imag
        
        # determine center of ellipsoid.
        if self.uniform:
            c     = np.zeros(g)
            intc  = np.zeros(g)
            leftc = np.zeros(g)
        else:
            c     = Yinv * y
            intc  = c.round()
            leftc = c - intc


        def find_integer_points(g, c, R, start):
            r"""
            Recursion function for computing the integer points needed in 
            each coordinate direction.

            INPUT:

            - ``T`` -- the Cholesky decomposition of the imaginary part of
              the Riemann matrix, `\Omega`
            
            - ``g`` -- the genus. recursively used to determine integer 
              points along each axis.

            - ``c`` -- center of integer point computation. `0 \in \CC^g` 
              is used when using the uniform approximation.

            - ``R`` -- the radius of the ellipsoid along the current axis.

            - ``start`` -- the starting integer point for each recursion 
              along each axis.

            OUTPUT:

            - ``intpoints`` -- (list) a list of all of the integer points 
              inside the bounding ellipsoid

            ... todo::

                Recursion can be memory intensive in Python. For genus `g<30`
                this is a reasonable computation but can be sped up by 
                writing a loop instead.
            """
            a = int(np.ceil((c[g] - R/T[g,g]).real))
            b = int(np.floor((c[g] + R/T[g,g]).real))

            # check if we reached the edge of the ellipsoid
            if not a < b: return []
            # last dimension reached: append points
            if g == 0: return [ [i] + start for i in range(a, b+1) ]
        
            #
            # compute new shifts, radii, start, and recurse
            #
            newg    = g-1
#            newT    = T.submatrix(nrows=newg+1, ncols=newg+1)
            newT    = T[:(newg+1),:(newg+1)]
            newTinv = la.inv(newT)
            pts     = []
            pdb.set_trace()
            for n in xrange(a, b+1):
                chat     = np.array(c[:newg+1])
                that     = np.array(T[:g][:newg+1])             # XXX COMPARE TO SAGE
                newc     = chat - newTinv * that * (n - c[g])
                newR     = np.sqrt(R**2/pi - (T[g,g] * (n - c[g]))**2)
                newstart = [n] + start
                pts     += find_integer_points(newg,newc,newR,newstart)

            return pts

        return find_integer_points(g-1, leftc, R, [])

    
    def radius(self, T, prec, deriv=[]):
        r"""
        Calculate the radius `R` to compute the value of the theta function
        to within `2^{-P + 1}` bits of precision where `P` is the 
        real / complex precision given by the input matrix. Used primarily
        by ``RiemannTheta.integer_points()``.

        `R` is the radius of [CRTF] Theorems 2, 4, and 6.

        INPUT:
        
        - ``T`` -- the Cholesky decomposition of the imaginary part of the 
          Riemann matrix `\Omega`

        - ``prec`` -- the desired precision of the computation
        
        - ``deriv`` -- (list) (default=``[]``) the derivative, if given. 
          Radius increases as order of derivative increases.

        EXAMPLES:

        Computing the radius. Note that the radius increases as a function of
        the precision::

            sage: R = ComplexField(10); I = R.gen()
            sage: Omega = matrix(R,2,2,[I,-1/2,-1/2,I])                  
            sage: theta = RiemannTheta(Omega)
            sage: theta.radius([])
            3.61513411073

            sage: R = ComplexField(40); I = R.gen()
            sage: Omega = matrix(R,2,2,[I,-1/2,-1/2,I])                  
            sage: theta = RiemannTheta(Omega)
            sage: theta.radius([])
            6.02254252538
            sage: theta.radius([[1,0]])
            6.37024100817
            
        """
        Pi = np.pi
        I  = 1.0j
        g  = T.shape[0]

        # compute the length of the shortest lattice vector
        U  = qflll(T)
        v  = (U*T)[:,0]
        r  = la.norm(v)
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

                INPUT:

                    - ``ins`` -- the quantity `(R-\rho)^2` where `R` is the radius we must solve for and `\rho` is the length of the shortest lattice vector in the integer lattice defined by `\Omega`.

                EXAMPLES:

                Since this function is used implicitly in 
                ``RiemannTheta.radius()`` we use an example input from above::

                    sage: R = ComplexField(40); I = R.gen()
                    sage: Omega = matrix(R,2,2,[I,-1/2,-1/2,I])
                    sage: theta = RiemannTheta(Omega)
                    sage: theta.radius([])
                    6.02254252538
                    sage: theta.radius([[1,0]])
                    6.37024100817
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



    def exp_and_osc_at_point(self, z, Omega, prec=1e-8, deriv=[]):
        r"""
        Calculate the exponential and oscillating parts of `\theta(z,\Omega)`.
        (Or a given directional derivative of `\theta`.) That is, compute 
        complex numbers `u,v \in \CC` such that `\theta(z,\Omega) = e^u v` 
        where the value of `v` is oscillatory as a function of `z`.

        INPUT:

        - ``z`` -- a list or tuple representing the complex `\CC^g` point at which to evaluate `\theta(z | \Omega)`

        - ``deriv`` -- (default: ``[]``) list representing the directional derivative of `\theta(z | \Omega)` you wish to compute

        OUTPUT:

        - ``(u,v)`` -- data pair such that `\theta(z,\Omega) = e^u v`.
            

        EXAMPLES:
        
        First, define a Riemann matrix and Riemann theta function::

            sage: from sage.functions.riemann_theta import RiemannTheta
            sage: R = ComplexField(36); I = R.gen()
            sage: Omega = matrix(R,2,2,[1.690983006 + 0.9510565162*I, 1.5 + 0.363271264*I, 1.5 + 0.363271264*I, 1.309016994+ 0.9510565162*I])
            sage: theta = RiemannTheta(Omega)

        Some example evaluations::

            sage: theta.exp_and_osc_at_point([0,0])
            (0, 1.050286258 - 0.1663490011*I)
            sage: theta.exp_and_osc_at_point([0.3+0.5*I,0.9+1.2*I])
            (4.763409165, 0.1568397231 - 1.078369835*I)
            sage: theta.exp_and_osc_at_point([0.3+0.5*I,0.9+1.2*I], deriv=[[1,0]])
            (4.763409165, -0.5864936847 + 0.04570614011*I)


        Defining a Riemann theta function, we demonstrate that the oscillatory
        part is periodic in each component with period 1::

            sage: theta.exp_and_osc_at_point([0,0])
            (0, 1.050286258 - 0.1663490011*I)
            sage: theta.exp_and_osc_at_point([1,3])
            (0, 1.050286258 - 0.1663490011*I)
            
        """
        g = Omega.shape[0]
        pi = np.pi

        # perform some simple cacheing on the matrices
        X = Omega.real
        Y = Omega.imag
        if (not self._Omega) or (self._Omega != Omega):
            # reset rad and intpoints since they're dependent on Omega
            self._rad       = None
            self._intpoints = None

            # define new matrix components
            self._Omega = Omega    
            self._Yinv  = la.inv(Y)
            self._T     = la.cholesky(Y)
            self._Tinv  = la.inv(self._T)
            
        # extract real and imaginary parts of input z
        z = np.array(z)
        x = z.real
        y = z.imag

        # convert derivatives to vector type       
        deriv = np.array(deriv)

        # compute integer points: check for uniform approximation
        if self.uniform:
            # check if we've already computed the uniform radius and intpoints
            if not self._rad:
                self._rad = self.radius(self._T, prec, deriv=deriv)
            if not self._intpoints:
                # fudge factor for uniform radius
                origin          = [0]*g
                self._intpoints = self.integer_points(self._Yinv, self._T, 
                                                      self._Tinv, origin, g, 
                                                      self._rad)
            R = self._rad
            S = self._intpoints
        else:
            R = self.radius(self._T, prec, deriv=deriv)
            S = self.integer_points(self._Yinv, self._T, self._Tinv,
                                    z, g, self._rad)

        # compute oscillatory and exponential terms
        v    = finite_sum(X, Y, self._T, x, y, S, deriv, domain)
        u    = pi*np.dot(y,self._Yinv * y)

        return u,v



    def value_at_point(self, z, Omega, deriv=[]):
        r"""
        Returns the value of `\theta(z,\Omega)` at a point `z`.
        
        INPUT:

        - ``z`` -- the complex `\CC^g` point at which to evaluate 
          `\theta(z,\Omega)`

        - ``deriv`` -- (default: ``[]``) list representing the directional 
          derivative of `\theta(z | \Omega)` you wish to compute


        OUTPUT:

            - `\theta(z | \Omega)` -- value of `\theta` at `z`


        EXAMPLES:
        
        Computing the value of a genus 2 Riemann theta function at the origin::

            sage: from sage.functions.riemann_theta import RiemannTheta
            sage: R = ComplexField(36); I = R.gen()
            sage: Omega = matrix(R,2,2,[1.690983006 + 0.9510565162*I, 1.5 + 0.363271264*I, 1.5 + 0.363271264*I, 1.309016994+ 0.9510565162*I])
            sage: theta = RiemannTheta(Omega)
            sage: theta.value_at_point([0,0])
            1.050286258 - 0.1663490011*I
            sage: theta([0,0])
            1.050286258 - 0.1663490011*I

        """
        exp_part, osc_part = self.exp_and_osc_at_point(z, Omega, deriv=deriv)
        return np.exp(exp_part) * osc_part

    def __call__(self, *args, **kwds):
        r"""
        Returns the value of `\theta(z,\Omega)` at a point `z`. Lazy evaluation
        is done if the input contains symbolic variables.
        
        INPUT:

        - ``z`` -- the complex `\CC^g` point at which to evaluate `\theta(z,\Omega)`

        - ``deriv`` -- (default: ``[]``) list representing the directional derivative of `\theta(z | \Omega)` you wish to compute


        OUTPUT:

            - `\theta(z | \Omega)` -- value of `\theta` at `z`


        EXAMPLES:
        
        Computing the value of a genus 2 Riemann theta function at the origin::

            sage: from sage.functions.riemann_theta import RiemannTheta
            sage: R = ComplexField(36); I = R.gen()
            sage: Omega = matrix(R,2,2,[1.690983006 + 0.9510565162*I, 1.5 + 0.363271264*I, 1.5 + 0.363271264*I, 1.309016994+ 0.9510565162*I])
            sage: theta = RiemannTheta(Omega)
            sage: theta([0,0])
            1.050286258 - 0.1663490011*I

        Performs lazy evaluation of symbolic input::

            sage: var('x')
            x
            sage: f = theta([x^2,sin(x)]); f
            theta(x^2, sin(x))
            sage: f(x=1.0*I)
            -94.35488925 - 59.48498251*I
            
        """
        return self.value_at_point(*args, **kwds)
        



if __name__=="__main__":
    print "=== Riemann Theta ==="
    theta = RiemannTheta()
    z = np.array([0,0])
    Omega = np.matrix([[1.0j,-0.5],[-0.5,1.0j]])

    print "Test #1:"
    print theta(z,Omega)
    print "1.1654 - 1.9522e-15*I"
    print 

    print "Test #2:"
    z = np.array([1.0j,1.0j])
    print theta(z)
    print -438.94 + 0.00056160*I
    print theta.exp_and_osc_at_point(z)
