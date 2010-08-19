"""
SPH functions for density and related computation.
"""

# local imports
from pysph.sph.sph_func cimport SPHFunctionParticle1D, SPHFunctionParticle2D, SPHFunctionParticle3D

###############################################################################
# `SPHRho3D` class.
###############################################################################
cdef class SPHRho3D(SPHFunctionParticle3D):
    """
    SPH function to compute density for 3d particles.
    """
    pass

cdef class SPHDensityRate3D(SPHFunctionParticle3D):
    """
    SPH function tom compute density rate for 3d particles.
    """
    pass
