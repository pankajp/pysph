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

cdef class MonaghanArtificialVsicosity(SPHFunctionParticle):
    """ MonaghanArtificialVsicosity """
    
    cdef public double gamma
    cdef public double alpha
    cdef public double beta
    cdef public double eta


cdef class MorrisViscosity(SPHFunctionParticle):
    """
    SPH function to compute pressure gradient.
    """
    cdef str mu
    cdef DoubleArray d_mu, s_mu