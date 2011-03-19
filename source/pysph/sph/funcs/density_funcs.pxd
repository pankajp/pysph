"""Declarations for the basic SPH functions 

"""

# Author: Prabhu Ramachandran <prabhu@aero.iitb.ac.in>
# Copyright (c) 2009, Prabhu Ramachandran

#sph imports
from pysph.sph.sph_func cimport SPHFunctionParticle, CSPHFunctionParticle

#base imports 
from pysph.base.particle_array cimport ParticleArray
from pysph.base.kernels cimport KernelBase

cdef class SPHRho(CSPHFunctionParticle):
    """
    SPH function to compute density for 3d particles.
    """
    pass

cdef class SPHDensityRate(SPHFunctionParticle):
    """
    SPH function tom compute density rate for 3d particles.
    """
    pass
