#base imports
from pysph.base.particle_array cimport ParticleArray
from pysph.base.kernels cimport KernelBase

cdef extern from "math.h":
    double pow(double x, double y)
    double sqrt(double x)

cdef class IdealGasEquation(SPHFunctionParticle):
    """ Ideal gas equation of state """
    
    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True, double gamma = 1.4):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays)
        self.gamma = gamma

        self.id = 'idealgas'
        self.tag = "state"

    cdef void eval(self, int k, int source_pid, int dest_pid,
                   KernelBase kernel, double *nr, double *dnr):
        """
        
        ::math::

        """
        cdef double Pa = self.d_p.data[dest_pid]
        cdef double ea = self.d_e.data[dest_pid]
        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double gamma = self.gamma

        nr[0] = (gamma - 1.0)*rhoa*ea
        nr[1] = sqrt(ea*(gamma-1))
        
        #nr[1] = sqrt(gamma*Pa/rhoa)
##############################################################################

cdef class TaitEquation(SPHFunctionParticle):
    """ Tait equation of state 
    
    The pressure is set as:

    P = B[(\frac{\rho}{\rho0})^gamma - 1.0]

    where,
    
    B = c0^2 \frac{\rho0}{\gamma}
    
    rho0 -- Reference density (default 1000)
    c0 -- sound speed at the reference density (10 * Vmax)
    Vmax -- estimated maximum velocity in the simulation
    gamma -- usually 7
    

    The sound speed is then set as
    
    cs = c0 * (\frac{\rho}{\rho0})^((gamma-1)/2)

    """
    
    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True, double co = 1.0,
                 double ro = 1000.0, double gamma=7.0, **kwargs):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays,
                                     **kwargs)
        self.co = co
        self.ro = ro
        self.gamma = gamma

        self.id = 'tait'
        self.tag = "state"

    cdef void eval(self, int k, int source_pid, int dest_pid,
                   KernelBase kernel, double *nr, double *dnr):
        """
        
        ::math::

        """
        cdef double rhoa, ratio, tmp, gamma2

        rhoa = self.d_rho.data[dest_pid]
        ratio = rhoa/self.ro

        gamma2 = 0.5*(self.gamma-1.0)

        tmp = pow(ratio, self.gamma)
        
        nr[0] = (tmp - 1.0)*self.co*self.co*self.ro/self.gamma
        nr[1] = pow(ratio, gamma2)*self.co
##############################################################################
