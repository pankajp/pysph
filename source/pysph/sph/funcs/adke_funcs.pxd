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
cdef class ADKEPilotRho(CSPHFunctionParticle):
    """ Compute the pilot estimate of density for the ADKE algorithm """
    cdef double h0

###############################################################################
# `ADKESmoothingUpdate` class.
###############################################################################
cdef class ADKESmoothingUpdate(ADKEPilotRho):
    """ Compute the new smoothing length for the ADKE algorithm """
    cdef double k, eps


###############################################################################
# `SPHDivergence` class.
###############################################################################
cdef class SPHVelocityDivergence(SPHFunctionParticle):
    """ Compute the pilot estimate of density for the ADKE algorithm """
    pass    

###############################################################################
# `ADKEConductionCoeffUpdate` class.
###############################################################################
cdef class ADKEConductionCoeffUpdate(SPHVelocityDivergence):
    """ Compute the new smoothing length for the ADKE algorithm """
    cdef double g1, g2
