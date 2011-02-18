""" Declarations for the adke functions """

#sph imports
from pysph.sph.sph_func cimport SPHFunctionParticle

#base imports 
from pysph.base.particle_array cimport ParticleArray
from pysph.base.point cimport Point
from pysph.base.kernels cimport KernelBase
from pysph.base.carray cimport DoubleArray


###############################################################################
# `PilotRho` class.
###############################################################################
cdef class PilotRho(SPHFunctionParticle):
    """ Compute the pillot estimate of density for the ADKE algorithm """

    cdef double h0

###############################################################################
# `SPHDivergence` class.
###############################################################################
cdef class SPHVelocityDivergence(SPHFunctionParticle):
    """ Compute the pillot estimate of density for the ADKE algorithm """

    pass
