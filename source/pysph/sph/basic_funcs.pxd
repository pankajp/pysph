"""
Module to hold some basic SPH functions.
"""

from pysph.base.carray cimport DoubleArray
from pysph.sph.sph_func cimport SPHFunctionParticle3D

cdef class SPH3D(SPHFunctionParticle3D):
    """
    Simple interpolation function for 3D cases.
    """
    cdef str prop_name
    cdef DoubleArray s_prop, d_prop

    cpdef int output_fields(self) except -1
