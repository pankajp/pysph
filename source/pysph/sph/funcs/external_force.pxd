"""Declarations for the External forces

"""

# Author: Prabhu Ramachandran <prabhu@aero.iitb.ac.in>
# Copyright (c) 2009, Prabhu Ramachandran

#sph imports
from pysph.sph.sph_func cimport SPHFunctionParticle

#base imports 
from pysph.base.particle_array cimport ParticleArray
from pysph.base.kernels cimport KernelBase
from pysph.base.point cimport Point, cPoint, cPoint_length

cdef class GravityForce(SPHFunctionParticle):
    """ MonaghanBoundaryForce """

    cdef public double gx, gy, gz
    
cdef class VectorForce(SPHFunctionParticle):
    """ MonaghanBoundaryForce """

    cdef public Point force

cdef class MoveCircleX(SPHFunctionParticle):
    """ Count Neighbors.  """

cdef class MoveCircleY(SPHFunctionParticle):
    """ Count Neighbors.  """
