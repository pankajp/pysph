""" Declarations for the adke functions """

#sph imports
from pysph.sph.sph_func cimport SPHFunctionParticle, CSPHFunctionParticle

#base imports 
from pysph.base.particle_array cimport ParticleArray
from pysph.base.point cimport cPoint
from pysph.base.kernels cimport KernelBase
from pysph.base.carray cimport DoubleArray, LongArray


###############################################################################
# `PilotRho` class.
###############################################################################
cdef class PilotRho(CSPHFunctionParticle):
    """ Compute the pilot estimate of density for the ADKE algorithm """
    cdef double h0

###############################################################################
# `SPHDivergence` class.
###############################################################################
cdef class SPHVelocityDivergence(SPHFunctionParticle):
    """ Compute the pilot estimate of density for the ADKE algorithm """
    pass

###############################################################################
# `ADKESmoothingUpdate` class.
###############################################################################
cdef class ADKESmoothingUpdate(PilotRho):
    """ Compute the new smoothing length for the ADKE algorithm """
    cdef double k, eps
