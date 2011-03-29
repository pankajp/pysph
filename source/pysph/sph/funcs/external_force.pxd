"""Declarations for the External forces

"""

# Author: Prabhu Ramachandran <prabhu@aero.iitb.ac.in>
# Copyright (c) 2009, Prabhu Ramachandran

#sph imports
from pysph.sph.sph_func cimport SPHFunction, SPHFunctionParticle

#base imports 
from pysph.base.particle_array cimport ParticleArray
from pysph.base.kernels cimport KernelBase
from pysph.base.point cimport Point, cPoint, cPoint_length

cdef class GravityForce(SPHFunction):
    """ MonaghanBoundaryForce """

    cdef public double gx, gy, gz
    
cdef class VectorForce(SPHFunction):
    """ MonaghanBoundaryForce """

    cdef public Point force

cdef class MoveCircleX(SPHFunction):
    """ Count Neighbors.  """

cdef class MoveCircleY(SPHFunction):
    """ Count Neighbors.  """

cdef class NBodyForce(SPHFunctionParticle):
    cdef public double eps
