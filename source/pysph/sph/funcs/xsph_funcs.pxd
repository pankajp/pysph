"""Declarations for the basic SPH functions 

"""

# Author: Prabhu Ramachandran <prabhu@aero.iitb.ac.in>
# Copyright (c) 2009, Prabhu Ramachandran

#sph imports
from pysph.sph.sph_func cimport SPHFunctionParticle

#base imports 
from pysph.base.particle_array cimport ParticleArray
from pysph.base.point cimport Point
from pysph.base.kernels cimport KernelBase
from pysph.base.carray cimport DoubleArray

cdef class XSPHCorrection(SPHFunctionParticle):
    """ The XSPH correction """
    cdef public double eps


cdef class XSPHDensityRate(SPHFunctionParticle):
    """ XSPHDensityRate function """
    
    cdef DoubleArray s_ubar, s_vbar, s_wbar
    cdef DoubleArray d_ubar, d_vbar, d_wbar
