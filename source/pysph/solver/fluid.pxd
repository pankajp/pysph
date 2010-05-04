"""
Module containing class to represent fluids.
"""

# local imports
from pysph.base.particle_array cimport ParticleArray

from entity_base cimport EntityBase
from entity_types cimport EntityTypes

cdef class Fluid(EntityBase):
    """
    """
    cpdef ParticleArray particle_array
