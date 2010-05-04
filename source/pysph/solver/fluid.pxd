"""
Module containing class to represent fluids.
"""

# local imports
from pysph.solver.entity_base cimport *

cdef class Fluid(EntityBase):
    """
    """
    cdef ParticleArray particle_array
