"""
Base class for all physical entities involved in the simulation.
"""
from pysph.base.particle_array cimport ParticleArray
from entity_types cimport EntityTypes

##############################################################################
# `EntityBase` class.
##############################################################################
cdef class EntityBase:
    """ Base class for any physical entity involved in a simulation."""

    # unique type identifier for the entity.
    cdef public int type

    # name of the entity
    cdef public str name

    # check if the entity is of type etype.
    cpdef bint is_a(self, int etype)

    # function to return the set of particles representing the entity.
    cpdef ParticleArray get_particle_array(self)

    # returns true if the entities type is included in the given list.
    cpdef bint is_type_included(self, list type_list)

#############################################################################
