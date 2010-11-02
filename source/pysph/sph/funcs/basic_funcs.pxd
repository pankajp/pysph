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

cdef class SPH(SPHFunctionParticle):
    """
    Simple interpolation function for 3D cases.
    """
    cdef public str prop_name
    cdef DoubleArray d_prop
    cdef DoubleArray s_prop

cdef class SPHGrad(SPHFunctionParticle):
    """
    SPH Gradient Approximation.
    """
    cdef public str prop_name
    cdef DoubleArray d_prop
    cdef DoubleArray s_prop

cdef class SPHSimpleDerivative(SPHFunctionParticle):
    """
    SPH Gradient Approximation.
    """
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


cdef class BonnetAndLokKernelGradientCorrectionTerms(SPHFunctionParticle):
    """ Kernel Gradient Correction terms """

cdef class FirstOrderCorrectionMatrix(SPHFunctionParticle):
    """ Kernel Gradient Correction terms """

cdef class FirstOrderCorrectionTermAlpha(SPHFunctionParticle):
    """ Kernel Gradient Correction terms """		
    cdef public str beta1, beta2, alpha, dbeta1dx, dbeta1dy
    cdef public str dbeta2dx, dbeta2dy	 

cdef class FirstOrderCorrectionMatrixGradient(SPHFunctionParticle):
    """ Kernel Gradient Correction terms """		
    
cdef class FirstOrderCorrectionVectorGradient(SPHFunctionParticle):
    """ Kernel Gradient Correction terms """		

