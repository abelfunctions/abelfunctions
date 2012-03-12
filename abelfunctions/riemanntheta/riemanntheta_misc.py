import numpy as np
import scipy.linalg as la
import warnings

try:
    import pyopencl as cl
except ImportError:
    warnings.warn("Could not find pyopencl package. Do not attempt to compute finite sum on GPU.")

def finite_sum_opencl(self, X, Y, Yinv, T, x, y, S, g):
    pass


def finite_sum(X, Y, Yinv, T, x, y, S, g, deriv):
    """
    Computes the oscillatory part of the finite sum
    
    .. math::
    
    \theta( z | \Omega ) = \sum_{ U_R } e^{ 2 \pi i \left( \tfrac{1}{2} \langle \Omega n, n \rangle +  \langle z, n \rangle \right) }
    
    where
    
    .. math::
    
        U_R = \left\{ n \in \ZZ^g : \pi ( n - c )^{t} \cdot Y \cdot 
        (n - c ) < R^2, |c_j| < 1/2, j=1,\ldots,g \right\}.
    
    The oscillatory part
    
    .. note::
    
        For accuracy issues we split the computation into its real and
        imaginary components.

    INPUT:
        
    - ``x`` -- the real part of the input vector `z`
        
    - ``y`` -- the imaginary part of the input vector `z`

    - ``S`` -- the set of integer points over which to compute the finite sum. 
      Often, this is set to `U_R` by calling ``RiemannTheta.integer_points()`.
        
    - ``deriv`` -- the derivative, if any

    - ``domain`` -- a ``RealField`` object. Sets the ring over which 
      comptuations are performed. For computational accuracy, we separate real 
      and imaginary parts of the finite sum and compute over the reals.

      
    OUTPUT:

    - the value of the oscillatory part of the Riemann theta function
            

    EXAMPLES:

    ``_finite_sum`` is implicitly called when, for example, computing the 
    value of a genus 2 Riemann theta function at the origin::

        sage: from sage.functions.riemann_theta import RiemannTheta
        sage: R = ComplexField(36); I = R.gen()
        sage: Omega = matrix(R,2,2,[1.690983006 + 0.9510565162*I, 1.5 + 0.363271264*I, 1.5 + 0.363271264*I, 1.309016994+ 0.9510565162*I])
        sage: theta = RiemannTheta(Omega)
        sage: theta.value_at_point([0,0])
        1.050286258 - 0.1663490011*I
    """    
    I     = 1.0j
    pi    = np.pi

    # define shifted vectors
    shift     = Yinv * y
    intshift  = shift.round()
    fracshift = shift - intshift

    # helper functions
    exppart  = lambda a: 2 * pi * np.dot((a-intshift).T, 0.5*X*(a-intshift)+x)
    normpart = lambda a: -pi*la.norm(T*(a+fracshift))**2

    if deriv:
        # for ease of computation, we perform derivative product
        # computation in a complex ring
        dd = [2*pi*I*np.dot(d,a-intshift) for d in deriv]
        derivprod = lambda a: np.prod(dd)
        
#    pdb.set_trace()       

    # compute the finite sum
    fsum_real = 0
    fsum_imag = 0
    for k in range(len(S)/g):
        n     = np.array(S[k*g:(k+1)*g]).reshape((g,1))
        ept   = exppart(n)
        npt   = np.exp(normpart(n))
        cpart = npt * np.cos(ept)
        spart = npt * np.sin(ept)
            
        if deriv:
            dp         = derivprod(n)
            dpr        = dp.real
            dpi        = dp.imag
            fsum_real += dpr*cpart - dpi*spart
            fsum_imag += dpi*cpart + dpr*spart
        else:
            fsum_real += cpart
            fsum_imag += spart

    return fsum_real + fsum_imag*1.0j

