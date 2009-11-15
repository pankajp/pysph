"""
Module to contain class to represent fluids.
"""

# local imports
from pysph.solver.entity_types cimport EntityTypes
from pysph.solver.entity_base cimport *

cdef class Fluid(EntityBase):
    """
    Base class to represent fluids.
    """
    def __cinit__(self, str name='', dict properties={}, 
                  ParticleArray particles=None, 
                  *args, **kwargs):
        """
        Constructor.
        """
        self.type = EntityTypes.Entity_Fluid
        self.particle_array = particles

        # create an empty particle array if nothing give from input.
        if self.particle_array is None:
            self.particle_array = ParticleArray()

        # add any default properties that are requiered of fluids in all kinds
        # of simulations.
        self.add_entity_property('rest_density', 1000.)
        self.add_entity_property('max_density_variation', 1.0)
        self.add_entity_property('actual_density_variation', 1.0)

    cpdef ParticleArray get_particle_array(self):
        """
        Returns the ParticleArray representing this entity.
        """
        return self.particle_array

    cpdef bint is_a(self, int type):
        """
        Check if this entity is of the given type.
        """
        return (EntityTypes.Entity_Fluid == type or
                EntityBase.is_a(self, type))
