"""
Base class for all physical entities involved in the simulation.
"""
#from pysph.base.attrdict import AttrDict
from pysph.base.carray cimport BaseArray
from pysph.base.particle_array cimport ParticleArray
from pysph.base.cell cimport CellManager


###############################################################################
# `EntityBase` class.
###############################################################################
cdef class EntityBase:
    """
    Base class for any physical entity involved in a simulation.    
    """
    # properties whose value is the same for the whole entity.
    cdef public properties
    
    # AttrDict (Bunch) for storing various information about the entity
    cdef public information 
    
    # unique type identifier for the entity.
    cdef public type type

    # name of the entity
    cdef public str name

    # additional components to that will modify properties of this solid.
    # this is will accessed by the integrator to add these components before the
    # appropriate stepper.
    cdef public dict modifier_components

    # add a property common to the whole entity
    cpdef add_entity_property(self, str prop_name, double default_value=*)

    # check if the entity is of type etype.
    cpdef bint is_a(self, type etype)

    # function to return the set of particles representing the entity.
    cpdef ParticleArray get_particle_array(self)

    # returns true if the entities type is included in the given list.
    cpdef bint is_type_included(self, list type_list)

    # adds a acceleration modified component.
    cpdef add_actuator(self, object actuator)

    # add more particles to the particle array.
    cpdef add_particles(self, ParticleArray parray, int group_id=*)

    # add all arrays that need to be binned for search to the cell manager.
    cpdef add_arrays_to_cell_manager(self, CellManager cell_manager)

