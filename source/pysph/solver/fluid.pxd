"""
Module containing class to represent fluids.
"""

# local imports
from pysph.base.particle_array cimport ParticleArray

from entity_base cimport EntityBase

cdef class Fluid(EntityBase):
    """
    """
    cpdef ParticleArray particle_array
