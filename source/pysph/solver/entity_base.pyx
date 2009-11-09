"""
Classes to represent any physical entity in a simulation.
"""

# logging imports
import logging
logger = logging.getLogger()

# local imports
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

    **Notes**

        INTEGRATION_PROPERTIES:
        By default, the INTEGRATION_PROPERTIES properties info is set to
        None. This says that the entity in itself does not have any specific
        requirements for integration. Its properties will be integrated
        depending on the settings of the integrator. To provide for specific
        integration requirements for this entity, add property names to this
        list in the information dict of the entity.

        TODO: A single place to refer to property names to be specified/used
        anywhere in the code.

    """
    # list of information keys provided by this object.
    INTEGRATION_PROPERTIES = 'INTEGRATION_PROPERTIES'
    
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
        
        self.information.set_list(self.INTEGRATION_PROPERTIES, None)

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

    def add_integration_property(self, str prop_name):
        """
        Adds a integration property requirement for use by an integrator class.
        """
        parray = self.get_particle_array()

        if parray is None:
            logger.warn('This entity does not provide a particle array')
            logger.warn('Not adding integration property')
            return
        
        ip = self.information.get_list(self.INTEGRATION_PROPERTIES)
        if ip is None:
            ip = []
            self.information.set_list(self.INTEGRATION_PROPERTIES, ip)
        ip.append(prop_name)

################################################################################
# `Fluid` class.
################################################################################
cdef class Fluid(EntityBase):
    """
    Base class to represent fluids.
    """
    def __cinit__(self, str name='', dict properties={}, dict particle_props={},
                  *args, **kwargs):
        """
        Constructor.
        """
        self.type = EntityTypes.Entity_Fluid
        self.particle_array = ParticleArray(name=self.name, **particle_props)

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

################################################################################
# `Solid` class.
################################################################################
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
