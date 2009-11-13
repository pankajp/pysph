"""
Contains various components to compute pressure gradients.
"""

# local imports
from pysph.solver.sph_component cimport SPHComponent

cdef class SymmetricPressureGradientComponent(SPHComponent):
    """
    Computes the pressure gradient using the SPHSymmetricPressureGradient3D
    function.
    """
    pass
    
