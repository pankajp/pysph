"""
SPH functions for pressure related computations.
"""

from pysph.base.carray cimport DoubleArray
from pysph.sph.sph_func cimport *

cdef class SPHSymmetricPressureGradient3D(SPHFunctionParticle3D):
    """
    SPH function to compute pressure gradient.
    """
    cdef str pressure
    cdef DoubleArray s_pressure, d_pressure
    
    cpdef int output_fields(self) except -1
