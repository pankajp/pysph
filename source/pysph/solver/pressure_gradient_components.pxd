"""
Contains various components to compute pressure gradients.
"""

# local imports
from pysph.solver.sph_component cimport SPHComponent

cdef class SPHSymmetricPressureGradientComponent(SPHComponent):
    """
    Computes the pressure gradient using the SPHSymmetricPressureGradient3D
    function.
    """
    cdef int compute(self) except -1
    cpdef int update_property_requirements(self) except -1
    
