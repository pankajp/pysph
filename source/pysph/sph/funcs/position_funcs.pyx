#cython: cdivision=True
#base imports 
from pysph.base.particle_array cimport ParticleArray
from pysph.base.kernels cimport KernelBase

###############################################################################
# `PositionStepping' class.
###############################################################################
cdef class PositionStepping(SPHFunction):
    """ Basic XSPH """

    #Defined in the .pxd file
    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True, hks=False):

        SPHFunction.__init__(self, source, dest, setup_arrays)

        self.id = 'positionstepper'
        self.tag = "position"

    cdef void eval_single(self, size_t dest_pid,
                          KernelBase kernel, double *result):
        """
        The expression used is:

        """
        result[0] = self.d_u.data[dest_pid]
        result[1] = self.d_v.data[dest_pid]
        result[2] = self.d_w.data[dest_pid]
        
##########################################################################
