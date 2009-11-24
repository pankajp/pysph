"""
Module containing components to compute viscosity.
"""

# local imports
from pysph.sph.sph_func cimport SPHFunctionParticle3D
from pysph.solver.sph_component cimport SPHComponent
from pysph.solver.speed_of_sound cimport SpeedOfSound

cdef class SPHMonaghanArtVisc3D(SPHFunctionParticle3D):
    """
    SPH Function to compute artificial viscosity.
    """
    cdef public double alpha
    cdef public SpeedOfSound speed_of_sound
    cdef public double epsilon

cdef class MonaghanArtViscComponent(SPHComponent):
    """
    Component to compute artificial viscosity.
    """
    cdef public double alpha
    cdef public double epsilon
    cdef public double beta

    cdef public SpeedOfSound speed_of_sound

    
