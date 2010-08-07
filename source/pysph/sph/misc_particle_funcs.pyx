"""
Miscellaneous SPHFunctionParticle's.
"""


# local imports
from pysph.base.kernels cimport KernelBase
from pysph.base.particle_array cimport ParticleArray
from pysph.sph.sph_func cimport SPHFunctionParticle

cdef class NeighborCountFunc(SPHFunctionParticle):
    """
    Class that counts the number of neighbors for each particle.
    """
    def __init__(self, ParticleArray source, ParticleArray dest, str h='h', str
                 mass='m', str rho='rho', setup_arrays=True):
        SPHFunctionParticle.__init__(self, source, dest, h, mass, rho, True)

    
    cdef void eval(self, int source_pid, int dest_pid, KernelBase kernel, double
                   *nr, double *dnr):
        """
        """
        nr[0] += 1.0

    cpdef int output_fields(self):
        """
        Returns the number of output fields, this SPHFunctionParticle will write
        to. This does not depend on the dimension of the simulation, it just
        indicates, the size of the arrays, dnr and nr that need to be passed to
        the eval function.
        """
        return 1

cdef class NeighborCountFunc2(SPHFunctionParticle):
    """
    Class that counts the number of neighbors of each particle.
    This class stores the count in two output fields.
    """
    def __init__(self, ParticleArray source, ParticleArray dest, str h='h', str
                 mass='m', str rho='rho', setup_arrays=True):
        SPHFunctionParticle.__init__(self, source, dest, h, mass, rho, True)

    cdef void eval(self, int source_pid, int dest_pid, KernelBase kernel, double
                   *nr, double *dnr):
        """
        """
        nr[0] += 1.0
        nr[1] += 1.0

    cpdef int output_fields(self):
        """
        """
        return 2

cdef class NeighborCountFunc3(SPHFunctionParticle):
    """
    Class that counts the number of neighbors of each particle.
    This class stores the count in three output fields.
    """
    def __init__(self, ParticleArray source, ParticleArray dest, str h='h', str
                 mass='m', str rho='rho', setup_arrays=True):
        SPHFunctionParticle.__init__(self, source, dest, h, mass, rho, True)

    cdef void eval(self, int source_pid, int dest_pid, KernelBase kernel, double
                   *nr, double *dnr):
        """
        """
        nr[0] += 1.0
        nr[1] += 1.0
        nr[2] += 1.0

    cpdef int output_fields(self):
        """
        """
        return 3

cdef class NeighborCountFunc4(SPHFunctionParticle):
    """
    Class that counts the number of neighbors of each particle.
    This class stores the count in four output fields.
    """
    def __init__(self, ParticleArray source, ParticleArray dest, str h='h', str
                 mass='m', str rho='rho', setup_arrays=True):
        SPHFunctionParticle.__init__(self, source, dest, h, mass, rho, True)

    cdef void eval(self, int source_pid, int dest_pid, KernelBase kernel, double
                   *nr, double *dnr):
        """
        """
        nr[0] += 1.0
        nr[1] += 1.0
        nr[2] += 1.0
        nr[3] += 1.0

    cpdef int output_fields(self):
        """
        """
        return 4
