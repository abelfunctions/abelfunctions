import numpy as np
import scipy.linalg as la
import warnings
import os.path

try:
    import pyopencl as cl

    # build context once. Use function below in finite_sum_opencl
    _platform = cl.get_platforms()[0]
    _device = [d for d in _platform.get_devices() if d.type==4][0]
    _context = cl.Context([_device])
    _queue = cl.CommandQueue(_context,
                             properties=cl.command_queue_properties.PROFILING_ENABLE)

    # load and build program
    basepath = os.path.dirname(__file__)
    filepath = os.path.abspath(os.path.join(basepath,'finite_sum_opencl.cl'))
    f = open(filepath,'r')
    fstr = "".join(f.readlines())
    f.close()
    _program = cl.Program(_context,fstr).build()
except ImportError:
    warnings.warn("Could not find pyopencl package. Do not attempt to compute finite sum on GPU.")



def finite_sum_opencl(X, Yinv, T, x, y, S, g):
    shift = Yinv*y
    intshift = shift.round().astype(np.float32)
    fracshift = (shift-intshift).astype(np.float32)

    # reshape data
    L = len(S)
    X = X.reshape((1,g*g)).astype(np.float32)
    T = T.reshape((1,g*g)).astype(np.float32)
    x = x.reshape((1,g)).astype(np.float32)
    S = S.reshape((1,L)).astype(np.float32)
    intshift = intshift.reshape((1,g))
    fracshift = fracshift.reshape((1,g))

    # create repository vectors
    fsum_real = np.empty((1,L/g)).astype(np.float32)
    fsum_imag = np.empty((1,L/g)).astype(np.float32)

    # create device memory buffers
    buf_X = cl.Buffer(_context,
                      cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                      hostbuf = X)
    buf_T = cl.Buffer(_context,
                      cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                      hostbuf = T)
    buf_x = cl.Buffer(_context,
                      cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                      hostbuf = x)
    buf_intshift = cl.Buffer(_context,
                      cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                      hostbuf = intshift)
    buf_fracshift = cl.Buffer(_context,
                      cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                      hostbuf = fracshift)
    buf_S = cl.Buffer(_context,
                      cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                      hostbuf = S)

    buf_fsum_real = cl.Buffer(_context,
                              cl.mem_flags.WRITE_ONLY,
                              fsum_real.nbytes)
    buf_fsum_imag = cl.Buffer(_context,
                              cl.mem_flags.WRITE_ONLY,
                              fsum_imag.nbytes)

    # call kernel and read output
    GLOBAL_SIZE = (L/g,)
    LOCAL_SIZE  = None
    event = _program.finite_sum_without_derivs(_queue, GLOBAL_SIZE, LOCAL_SIZE,
                                               buf_X,
                                               buf_T,
                                               buf_x,
                                               buf_intshift,
                                               buf_fracshift,
                                               np.int32(g),
                                               buf_S,
                                               buf_fsum_real,
                                               buf_fsum_imag)

    
    cl.enqueue_copy(_queue,fsum_real,buf_fsum_real)
    cl.enqueue_copy(_queue,fsum_imag,buf_fsum_imag)

    # compute finite sum
    return np.sum(fsum_real) + 1.0j*np.sum(fsum_imag)

def finite_sum(X, Yinv, T, x, y, S, g, deriv):
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
    shift     = Yinv*y
    intshift  = shift.round()
    fracshift = shift - intshift

    # helper functions
    exppart  = lambda a: 2 * pi * np.dot((a-intshift).T, 0.5*X*(a-intshift)+x)
    normpart = lambda a: -pi*la.norm(T*(a+fracshift))**2

    if (len(deriv) > 0):
        # for ease of computation, we perform derivative product
        # computation in a complex ring
        derivprod = lambda a: np.prod([2*pi*I*np.dot(d,a-intshift) for d in deriv])
        
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
            
        if (len(deriv) > 0):
            dp         = derivprod(n)
            dpr        = dp.real
            dpi        = dp.imag
            print(str(dpr) +" "+ str(dpi))
            fsum_real += dpr*cpart - dpi*spart
            fsum_imag += dpi*cpart + dpr*spart
        else:
            fsum_real += cpart
            fsum_imag += spart

    return fsum_real + fsum_imag*1.0j

