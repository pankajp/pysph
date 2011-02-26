"""
SPH functions for density and related computation.
"""

#sph imports
from pysph.sph.sph_func cimport SPHFunctionPoint

#base imports 
from pysph.base.particle_array cimport ParticleArray
from pysph.base.point cimport cPoint
from pysph.base.kernels cimport KernelBase
from pysph.base.carray cimport DoubleArray


cdef class SPHEval(SPHFunctionPoint):
    """ Simple SPH interpolation class """

    cdef public str prop_name
    cdef DoubleArray s_prop

cdef class SPHSimpleDerivativeEval(SPHFunctionPoint):
    """
    SPH Gradient Approximation.
    """
    cdef public str prop_name
    cdef DoubleArray s_prop

cdef class CSPMEval(SPHFunctionPoint):
    """ CSPM Interpolation of a function"""

    cdef public str prop_name
    cdef DoubleArray s_prop

cdef class CSPMDerivativeEval(SPHFunctionPoint):
    """ CSPM Interpolation of a function"""

    cdef public str prop_name
    cdef DoubleArray s_prop


