"""Declarations for the basic SPH functions 

"""

# Author: Prabhu Ramachandran <prabhu@aero.iitb.ac.in>
# Copyright (c) 2009, Prabhu Ramachandran

#sph imports
from pysph.sph.sph_func cimport SPHFunctionParticle

#base imports 
from pysph.base.particle_array cimport ParticleArray
from pysph.base.point cimport cPoint, cPoint_sub, cPoint, cPoint_length
from pysph.base.kernels cimport KernelBase
from pysph.base.carray cimport DoubleArray

cdef class MonaghanBoundaryForce(SPHFunctionParticle):
    """ MonaghanBoundaryForce """

    cdef public double delp
    
    cdef public DoubleArray s_tx, s_ty, s_tz, s_nx, s_ny, s_nz

cdef class BeckerBoundaryForce(SPHFunctionParticle):
    """
    Class to compute the interaction of a boundary particle on a fluid 
    particle.
    """
    cdef public double sound_speed

cdef class LennardJonesForce(SPHFunctionParticle):
    """
    Class to compute the interaction of a boundary particle on a fluid 
    particle.
    """

    cdef public double D
    cdef public double ro
    cdef public double p1, p2
