"""
Classes to represent any physical entity in a simulation.
"""

# logging imports
import logging
logger = logging.getLogger()

# standard imports
import types

# local imports
from pysph.base.attrdict import AttrDict
from pysph.base.carray cimport BaseArray
from pysph.base.particle_array cimport ParticleArray
from pysph.base.cell cimport CellManager


###############################################################################
# `EntityBase` class.
###############################################################################
cdef class EntityBase:
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
        self.type = EntityBase

        self.name = name

        # set the properties.
        self.properties = dict()
        self.properties.update(properties)
        
        self.information = AttrDict()
        self.information[self.INTEGRATION_PROPERTIES] = None

    cpdef add_entity_property(self, str prop_name, double default_value=0.0):
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

    cpdef bint is_a(self, type etype):
        """
        Check if this entity is of the given type.

        This will be implemented differently in the derived classes.
        """
        if self.type == etype or isinstance(self, (etype,)):
            return True
        else:
            return False

    def set_properties_to_integrate(self, list prop_names):
        """
        Adds a integration property requirement for use by an integrator class.
        """
        parray = self.get_particle_array()

        if parray is None:
            logger.warn('This entity does not provide a particle array')
            logger.warn('Not adding integration property')
            return
        
        cdef list ip = self.information[self.INTEGRATION_PROPERTIES]
        if ip is None:
            ip = []
            self.information[self.INTEGRATION_PROPERTIES] = ip

        ip.extend(list(set(prop_names)))

    cpdef bint is_type_included(self, list types):
        """
        Returns true if the entity's type or any of its parent types is included
        in the the list of types passed.
        """
        for e_type in types:
            if self.is_a(e_type):
                return True

        return False

    cpdef add_actuator(self, object actuator):
        """
        Adds an acceleration modifier to the entity.
        """
        cdef list accel_modifier = self.modifier_components['acceleration']
        
        if accel_modifier is None:
            accel_modifier = []
            self.modifier_components['acceleration'] = accel_modifier

        if accel_modifier.count(actuator) == 0:
            accel_modifier.append(actuator)

    cpdef add_particles(self, ParticleArray particles, int group_id=0):
        """
        Adds the given particles into the entities particle array. This by
        default adds the particles to the particle array returned by the
        get_particle_array call.

        """
        parray = self.get_particle_array()
        if parray is None:
            logger.warn('No particle array returned by %s'%(self.name))
            return

        # first set the group id of the new particles if requested.
        if group_id != 0:
            group = particles.get('group', only_real_particles=False)
            group[:] = group_id
            
        parray.append_parray(particles)        

    cpdef add_arrays_to_cell_manager(self, CellManager cell_manager):
        """
        Add all arrays that need to be binned for this entity to the cell manager.
        """
        raise NotImplementedError, 'EntityBase::add_arrays_to_cell_manager'
