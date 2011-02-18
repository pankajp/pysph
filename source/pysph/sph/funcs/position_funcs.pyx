#base imports 
from pysph.base.particle_array cimport ParticleArray
from pysph.base.kernels cimport KernelBase

###############################################################################
# `PositionStepping' class.
###############################################################################
cdef class PositionStepping(SPHFunctionParticle):
    """ Basic XSPH """

    #Defined in the .pxd file
    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True):
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays)

        self.id = 'positionstepper'
        self.tag = "position"

    cdef void eval(self, int k, int source_pid, int dest_pid,
                   KernelBase kernel, double *nr, double *dnr):
        """
        The expression used is:

        """
        nr[0] = self.d_u.data[dest_pid]
        nr[1] = self.d_v.data[dest_pid]
        nr[2] = self.d_w.data[dest_pid]
        
##########################################################################
