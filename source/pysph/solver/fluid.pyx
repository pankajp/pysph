"""
Module to contain class to represent fluids.
"""

cdef class Fluid(EntityBase):
    """
    Base class to represent fluids.
    """
    def __cinit__(self, str name='', 
                  ParticleArray particles=None, 
                  *args, **kwargs):
        """
        Constructor.
        """
        self.type = EntityTypes.Entity_Fluid
        self.particle_array = particles

        # create an empty particle array if nothing give from input.
        if self.particle_array is None:
            self.particle_array = ParticleArray(name=self.name)
        
        # name of particle array same as name of entity.
        self.particle_array.name = self.name

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

##########################################################################


