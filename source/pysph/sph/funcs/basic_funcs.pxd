"""Declarations for the basic SPH functions 

"""

# Author: Prabhu Ramachandran <prabhu@aero.iitb.ac.in>
# Copyright (c) 2009, Prabhu Ramachandran

#sph imports
from pysph.sph.sph_func cimport SPHFunction, SPHFunctionParticle, CSPHFunctionParticle

#base imports 
from pysph.base.particle_array cimport ParticleArray
from pysph.base.point cimport cPoint, cPoint_dot, cPoint_length, cPoint_sub
from pysph.base.kernels cimport KernelBase
from pysph.base.carray cimport DoubleArray

cdef class SPH(CSPHFunctionParticle):
    """
    Simple interpolation function for 3D cases.
    """
    cdef public str prop_name
    cdef DoubleArray d_prop
    cdef DoubleArray s_prop

cdef class SPHSimpleGradient(SPHFunctionParticle):
    """
    SPH Gradient Approximation.
    """
    cdef public str prop_name
    cdef DoubleArray d_prop
    cdef DoubleArray s_prop

cdef class SPHGradient(SPHFunctionParticle):
    """ SPH Gradient Approximation """
    cdef public str prop_name
    cdef DoubleArray d_prop
    cdef DoubleArray s_prop

cdef class SPHLaplacian(SPHFunctionParticle):
    """ SPH Laplacian estimation """
    cdef public str prop_name
    cdef DoubleArray d_prop
    cdef DoubleArray s_prop

cdef class CountNeighbors(SPHFunctionParticle):
    """ Count Neighbors.  """
    pass

cdef class BonnetAndLokKernelGradientCorrectionTerms(CSPHFunctionParticle):
    """ Kernel Gradient Correction terms """
    pass

cdef class FirstOrderCorrectionMatrix(CSPHFunctionParticle):
    """ Kernel Gradient Correction terms """
    pass

cdef class FirstOrderCorrectionTermAlpha(SPHFunctionParticle):
    """ Kernel Gradient Correction terms """		
    cdef public str beta1, beta2, alpha, dbeta1dx, dbeta1dy
    cdef public str dbeta2dx, dbeta2dy	 

cdef class FirstOrderCorrectionMatrixGradient(CSPHFunctionParticle):
    """ Kernel Gradient Correction terms """
    pass		
    
cdef class FirstOrderCorrectionVectorGradient(CSPHFunctionParticle):
    """ Kernel Gradient Correction terms """
    pass		
