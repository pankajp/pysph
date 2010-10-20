#base imports
from pysph.base.particle_array cimport ParticleArray
from pysph.base.kernels cimport MultidimensionalKernel

cdef extern from "math.h":
    double pow(double x, double y)

cdef class IdealGasEquation(SPHFunctionParticle):
    """ Ideal gas equation of state """
    
    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True, double gamma = 1.4):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays)
        self.gamma = gamma
        self.id = 'idealgas'

    cdef void eval(self, int source_pid, int dest_pid,
                   MultidimensionalKernel kernel, double *nr, double *dnr):
        """
        
        ::math::

        """
        cdef double ea = self.d_e.data[dest_pid]
        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double gamma = self.gamma

        nr[0] = (gamma - 1.0)*rhoa*ea
##############################################################################

cdef class TaitEquation(SPHFunctionParticle):
    """ Tait equation of state """
    
    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True, double ko = 1.0, 
                 double ro = 1000.0, double gamma=7):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays)
        self.ko = ko
        self.ro = ro
        self.gamma = gamma
        self.id = 'tait'

    cdef void eval(self, int source_pid, int dest_pid,
                   MultidimensionalKernel kernel, double *nr, double *dnr):
        """
        
        ::math::

        """
        cdef double rhoa, ratio, tmp

        rhoa = self.d_rho.data[dest_pid]
        ratio = rhoa/self.ro

        tmp = pow(ratio, self.gamma)
        
        nr[0] = (tmp - 1.0)*self.ko
##############################################################################
