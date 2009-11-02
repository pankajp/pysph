"""
Classes to represent any physical entity in a simulation.
"""

from pysph.base.carray cimport BaseArray
from pysph.base.particle_array cimport ParticleArray

from pysph.solver.base cimport Base
from pysph.solver.entity_types cimport EntityTypes


################################################################################
# `EntityBase` class.
################################################################################
cdef class EntityBase(Base):
    """
    Base class for any physical entity involved in a simulation.
    """
    # list of information keys provided by this object.
    INTEGRABLE_PROPERTIES = 'INTEGRABLE_PROPERTIES'
    
    def __cinit__(self, str name ='', dict properties={}, *args, **kwargs):
        """
        Constructor.
        """
        # set type to base.
        self.type = EntityTypes.Entity_Base

        self.name = name

        # set the properties.
        self.properties = {}
        self.properties.update(properties)

    cpdef add_property(self, str prop_name, double default_value=0.0):
        """
        Add a physical property for the entity which is common to the whole
        entity.
        """
        if self.properties.has_key(prop_name):
            return
        else:
            self.properties[prop_name] = default_value

    cpdef ParticleArray get_particle_array(self):
        """
        Return the particle_array representing this entity.

        Reimplement this in the concrete entity types, which may or may not have
        particle representation of themselves.

        """
        return None

    cpdef bint is_a(self, int type):
        """
        Check if this entity is of the given type.

        This will be implementde differently in the derived classes.
        """
        if EntityTypes.Entity_Base == type:
            return True


################################################################################
# `Fluid` class.
################################################################################
cdef class Fluid(EntityBase):
    """
    Base class to represent fluids.
    """
    def __cinit__(self, str nam='', dict properties={}, dict particle_props={}, *args,
                  **kwargs):
        """
        Constructor.
        """
        self.type = EntityTypes.Entity_Fluid
        self.particle_array = ParticleArray(particle_props)

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


cdef class Solid(EntityBase):
    """
    Base class to represent solids.
    """
    def __cinit__(self, str name='', dict properties={}, *args, **Kwargs):
        """
        Constructor.
        """
        self.type = EntityTypes.Entity_Solid
        
    cpdef bint is_a(self, int type):
        """
        Check if this entity is of the given type.
        """
        return (EntityTypes.Entity_Solid == type or
                EntityBase.is_a(self, type))
