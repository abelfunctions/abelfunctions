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

It is holomorphic in both `z` y and `\Omega`. It is quasiperiodic in `z`
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

- Grady Williams (2012-2013)

"""
import numpy as np
import scipy as sp
import scipy.linalg as la
import riemanntheta_cy
from scipy.special import gamma, gammaincc, gammainccinv,gammaincinv
from scipy.optimize import fsolve
import time
from lattice_reduction import lattice_reduce
from siegel import siegel

#For testing purposes only
from lattice_plotter import *

gpu_capable = True
try:
    from riemanntheta_cuda import RiemannThetaCuda
except ImportError:
    gpu_capable = False


class RiemannTheta_Function(object):
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
        self._prec      = 1e-8
        if (gpu_capable):
            self.parRiemann = RiemannThetaCuda(32, 16) 

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


    def find_int_points(self,g, c, R, T,start):
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
        print c
        print g
        a_ = c[g] - R/(np.sqrt(np.pi)*T[g,g]) 
        b_ = c[g] + R/(np.sqrt(np.pi)*T[g,g])
        a = np.ceil(a_)
        b = np.floor(b_)
        # check if we reached the edge of the ellipsoid
        if not a <= b: return np.array([])
        # last dimension reached: append points
        if g == 0:
            points = np.array([])
            for i in range(a, b+1):
                #Note that this algorithm works backwards on the coordinates,
                #the last coordinate found is x1 if our coordinates are {x1,x2, ... xn}
                points = np.append(np.append([i],start), points)
            return points
        #
        # compute new shifts, radii, start, and recurse
        #
        newg = g-1
        newT = T[:(newg+1),:(newg+1)]
        newTinv = la.inv(newT)
        pts = []
        for n in range(a, b+1):
            chat = c[:newg+1]
            that = T[:newg+1,g]
            newc = (chat.T - (np.dot(newTinv, that)*(n - c[g]))).T
            newR = np.sqrt(R**2 - np.pi*(T[g,g] * (n - c[g]))**2) # XXX
            newstart = np.append([n],start)
            newpts = self.find_int_points(newg,newc,newR,newT,newstart)
            pts = np.append(pts,newpts)
        return pts


    def integer_points(self, Yinv, T, Tinv, z, g, R):
        """
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
       # g    = Yinv.shape[0]
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
        int_points = self.find_int_points(g-1,leftc,R,T,[])
        return int_points

    def test(self):
        print "WTF"

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
	A = lattice_reduce(T)
        r  = min(la.norm(A[:,i]) for i in range(int(g)))
        normTinv = la.norm(la.inv(T))

        # solve for the radius using:
        #   * Theorem 3 of [CRTF] (no derivative)
        #   * Theorem 5 of [CRTF] (first order derivative)
        #   * Theorem 7 of [CRTF] (second order derivative
        if len(deriv) == 0:
            eps  = prec
            lhs  = eps * (2.0/g) * (r/2.0)**g * gamma(g/2.0)
            ins  = gammainccinv(g/2.0,lhs)
            R    = np.sqrt(ins) + r/2.0
            rad  = max( R, (np.sqrt(2*g)+r)/2.0)
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
            prodnormderiv = np.prod([la.norm(d) for d in deriv])

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

    """
    Performs simple recacheing of matrices, also prepares gpu for processing if necessary
    """
    def recache(self, Omega, X, Y, Yinv, T, g, prec, deriv, Tinv):
        recache_omega = not np.array_equal(self._Omega, Omega)
        recache_prec = self._prec != prec
        # check if we've already computed the uniform radius and intpoints
        if (recache_omega or recache_prec):
            self._prec = prec
            self._rad = self.radius(T, prec, deriv=deriv)
            origin = [0]*g
            self._intpoints = self.integer_points(Yinv, T, Tinv, origin, 
                                                  g, self._rad)
        if (gpu_capable):
            self.parRiemann.cache_intpoints(self._intpoints)
            if (self._Omega is None or not g == self._Omega.shape[0]):
                self.parRiemann.compile(g)
                self.parRiemann.cache_omega_real(X)
                self.parRiemann.cache_omega_imag(Yinv, T)
            else:
                if (not np.array_equal(self._Omega.real, Omega.real)):
                    self.parRiemann.cache_omega_real(X)
                if (not np.array_equal(self._Omega.imag, Omega.imag)):
                    self.parRiemann.cache_omega_imag(Yinv, T)
        self._Omega = Omega

    """
    Handles gpu processing of data sets which are too large to fit into the memory of the gpu
    at once
    """
    def gpu_process(self, Z, deriv, gpu_max, length):
        v = np.array([])
        u = np.array([])
        #divide the set z into as many partitions as necessary
        num_partitions = (length-1)//(gpu_max) + 1
        for i in range(0, num_partitions):
            #determine the starting and stopping points of the partition
            p_start = (i)*gpu_max
            p_stop = min(length, (i+1)*gpu_max)
            if (len(deriv) > 0):
                v_p = self.parRiemann.compute_v_with_derivs(Z[p_start: p_stop, :], deriv)
            else:
                v_p = self.parRiemann.compute_v_without_derivs(Z[p_start: p_stop, :])
            u_p = self.parRiemann.compute_u()
            u = np.concatenate((u, u_p))
            v = np.concatenate((v, v_p))
        return u,v
 
    

    def exp_and_osc_at_point(self, z, Omega, batch = False, prec=1e-9, deriv=[], gpu=gpu_capable, gpu_max = 500000):
        r"""
        Calculate the exponential and oscillating parts of `\theta(z,\Omega)`.
        (Or a given directional derivative of `\theta`.) That is, compute 
        complex numbers `u,v \in \CC` such that `\theta(z,\Omega) = e^u v` 
        where the value of `v` is oscillatory as a function of `z`.            
        """
        g = Omega.shape[0]
        pi = np.pi

        #Process all of the matrices into numpy matrices
        X = np.array(Omega.real)
        Y = np.array(Omega.imag)
        Yinv = np.array(la.inv(Y))
        T = np.array(la.cholesky(Y))
        Tinv = np.array(la.inv(T))
        deriv = np.array(deriv)
        
        #Do recacheing if necessary
        self.recache(Omega, X, Y, Yinv, T, g, prec, deriv, Tinv)

        # extract real and imaginary parts of input z
        length = 1
        if batch:
            length = len(z)
        z = np.array(z).reshape((length, g))
        # compute integer points: check for uniform approximation
        if self.uniform:
            R = self._rad
            S = self._intpoints
        elif(batch):
                raise Exception("Can't compute pointwise approximation for multiple points.\nUse uniform approximation or call the function seperately for each point.")
        else:
            R = self.radius(T, prec, deriv=deriv)
            S = self.integer_points(Yinv, T, 
Tinv, z, g, R)
        # compute oscillatory and exponential terms
        if gpu and (length > gpu_max):
            u,v = self.gpu_process(z, deriv, gpu_max, length)
        elif gpu and batch and len(deriv) > 0:
            v = self.parRiemann.compute_v_with_derivs(z, deriv)
        elif gpu and batch:
            v = self.parRiemann.compute_v_without_derivs(z)
        elif (len(deriv) > 0):
            v = riemanntheta_cy.finite_sum_derivatives(X, Yinv, T, z, S, deriv, g, batch)
        else:
            v = riemanntheta_cy.finite_sum(X, Yinv, T, z, S, g, batch)
        if (length > gpu_max):
            #u already computed
            pass
        elif (gpu and batch):
            u = self.parRiemann.compute_u()
        elif (batch):
            K = len(z)
            u = np.zeros(K)
            for i in range(K):
                w = np.array([z[i,:].imag])
                val = pi*np.dot(w, Yinv*w.T).item(0,0)
                u[i] = val
        else:
            u = np.pi*np.dot(z.imag,np.dot(Yinv,z.imag.T)).item(0,0)
            print "The value of U:"
            print u
        return u,v

    def siegel_exp_and_osc(self, z, Omega, prec=1e-8, deriv=[],gpu=gpu_capable,batch=False):
        g = Omega.shape[0]
        pi = np.pi
        z = np.array(z, dtype=np.complex)

        #Process all of the matrices into numpy arrays
        Omega = np.array(Omega, dtype=np.complex)
        #Perform the siegel transformation
        Om, mod = siegel(Omega, g)
        a = mod[:g, :g].astype(np.double)
        b = mod[:g, g:].astype(np.double)
        c = mod[g:, :g].astype(np.double)
        d = mod[g:, g:].astype(np.double)
        print "Returned Omega"
        print Om
        print
        print "Calculated Omega"
        print np.dot((np.dot(a,Omega) + b), la.inv(np.dot(c,Omega) + d))
        print
        C_orig = np.dot(c,Omega)+d
        C = la.inv(np.dot(c,Omega)+d)
        new_z = np.dot(C, z)
        u,v = self.exp_and_osc_at_point(new_z,Om)
        determinant_part = np.sqrt(la.det(C_orig))
        print "shift:"
        shift_part = np.pi*1.0j*np.dot(np.dot(z,C),np.dot(c,z))
        print shift_part
        return (1.0/(determinant_part))*np.exp(-shift_part)*v
        
    def characteristic(self, chars, z, Omega):
        alpha, beta = chars[0], chars[1]
        shift = np.dot(Omega, alpha) + beta
        z = z + shift
        u,v = self.exp_and_osc_at_point(z, Omega)
        exp_shift = 2*np.pi*1.0j*(.5*np.dot(alpha,np.dot(Omega,alpha)) + np.dot(alpha,np.dot(Omega,alpha)))
        return np.exp(u - exp_shift)*v

         
    def value_at_point(self, z, Omega, prec=1e-8, deriv=[], gpu=gpu_capable, batch=False):
        r"""
        Returns the value of `\theta(z,\Omega)` at a point `z` or set of points if batch is True.
        """
        
        exp_part, osc_part = self.exp_and_osc_at_point(z, Omega, prec=prec,
                                                       deriv=deriv, gpu=gpu,batch=batch)
        
        return np.exp(exp_part) * osc_part

    def __call__(self, z, Omega, prec=1e-8, deriv=[], gpu=gpu_capable, batch=False):
        r"""
        Returns the value of `\theta(z,\Omega)` at a point `z`. Lazy evaluation
        is done if the input contains symbolic variables. If batch is set to true
        then the functions expects a list/numpy array as input and returns a numpy array as output
        """
        return self.value_at_point(z, Omega, prec=prec, deriv=deriv, gpu=gpu, batch=batch)
                


# declaration of Riemann theta
RiemannTheta = RiemannTheta_Function()
        

if __name__=="__main__": 
    print "=== Riemann Theta ==="
    theta = RiemannTheta
    z = np.array([0,0])
    Omega = np.matrix([[1.0j,-0.5],[-0.5,1.0j]])
    
    print "Test #1:"
    print theta.value_at_point(z,Omega)
    print "1.1654 - 1.9522e-15*I"
    print 
    print theta._intpoints
    print "Test #2:"
    z1 = np.array([1.0j,1.0j])
    print theta.value_at_point(z1,Omega)
    print "-438.94 + 0.00056160*I"
    print

    print "Batch Test"
    z0 = np.array([0, 0])
    z1 = np.array([1.0j,1.0j])
    z2 = np.array([.5 + .5j, .5 + .5j])
    z3 = np.array([0 + .5j, .33 + .8j])
    z4 = np.array([.345 + .768j, -44 - .76j])
    print theta.value_at_point([z0,z1,z2,z3,z4],Omega, batch=True)
    print
    
    if (gpu_capable):
        a = []
        for x in range(200000):
            a.append(z0)
            a.append(z1)
            a.append(z2)
            a.append(z3)
            a.append(z4)
        start1 = time.clock()
        print theta.value_at_point(a, Omega, batch=True, prec=1e-12)[57000:57005]
        print("GPU time to perform calculation: " + str(time.clock() - start1))
        start2 = time.clock()
        #print theta.value_at_point(a, Omega, gpu=False, batch=True,prec=1e-12)[1730:1735]
        #print("CPU time to do same calculation: " + str(time.clock() - start2))

    print
    print "Derivative Tests:"
    print "Calculating directional derivatives at z = [i, 0]"
    print
    y = np.array([1.0j, 0])
    print "For [[1,0]]:"
    print theta.value_at_point(y, Omega, deriv = [[1,0]])
    print "0 - 146.49i"
    print
    print "For [[1,0] , [0,1]]: "
    print theta.value_at_point(y, Omega, deriv = [[1,0], [0,1]])
    print "0 + 0i" 
    print
    print "For [[0,1], [1,0]]: "
    print theta.value_at_point(y, Omega, deriv = [[0,1], [1,0]])
    print "0 + 0i"
    print
    print "For [[1,0],[1,0],[1,1]]:"
    print theta.value_at_point(y, Omega, deriv = [[1,0], [1,0], [1,1]])
    print "0 + 7400.39i" 
    print
    print "For [[1,1],[1,1],[1,1],[1,1]]: "
    print theta.value_at_point(y, Omega, deriv = [[1,1],[1,1],[1,1],[1,1]])
    print "41743.92 + 0i" 
    print
    print ("GPU Derivative Test")
    l = []
    for x in range(5):
        l.append(y)
    print theta.value_at_point(l, Omega, deriv = [[1,1],[1,1],[1,1],[1,1]], batch=True)
    
   
    print "Test #3"
    import pylab as p
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt

    print "\tCalculating theta..."
    SIZE = 60
    x = np.linspace(0,1,SIZE)
    y = np.linspace(0,5,SIZE)
    X,Y = p.meshgrid(x,y)
    Z = X + Y*1.0j
    Z = Z.flatten()
    U,V = theta.exp_and_osc_at_point([[z,0] for z in Z], Omega, batch=True)
    Z = (V.reshape(60,60)).imag
    print "\tPlotting..."
    plt.contourf(X,Y,Z,7,antialiased=True)
    plt.show()

    SIZE = 100
    x = np.linspace(-7,7,SIZE)
    y = np.linspace(-7,7,SIZE)
    X,Y = p.meshgrid(x,y)
    Z = X + Y*1.j
    Z = X + Y*1.j
    Z = Z.flatten()
    w = np.array([[1.j]])
    print w
    U,V = theta.exp_and_osc_at_point(Z, w, batch = True)
    print theta._intpoints
    Z = (V.reshape(100,100)).real
    plt.contourf(X,Y,Z,7,antialiased=True)
    plt.show()

    omega = np.identity(3)*1.j
    z = np.array([0,0,0])
    theta.value_at_point(z, omega)
    """
    print "Siegel Test"
    Omega = -1.0/(2 * np.pi * 1.0j) * np.array([[111.207, 96.616], [96.616, 83.943]],dtype=np.complex)
    Om = 1.j*np.array([[8,7],[7,8]])
    print Om
    #print "Determinant of Omega"
    #print la.det(Omega)
    #print "-----------------"
    x = np.array([0,0])
    print "Calculating original"
    val_orig = theta.exp_and_osc_at_point(x,Om, prec = 1e-12)[1]
    print val_orig
    print "Calculating Siegel"
    val_sigel = theta.siegel_exp_and_osc(x,Om, prec = 1e-12)
    print val_sigel
    print "Absolute Error"
    print np.abs(val_orig - val_sigel)
    """
