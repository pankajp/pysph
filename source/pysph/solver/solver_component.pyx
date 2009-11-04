"""
Base class for all solver components.
"""

# logger imports
import logging
logger = logging.getLogger()

# local imports
from pysph.solver.base cimport Base
from pysph.solver.entity_base cimport EntityBase

from pysph.solver.typed_dict cimport TypedDict


################################################################################
# `SolverComponent` class.
################################################################################
cdef class SolverComponent(Base):
    """
    Base class for all components.

    **Notes**

        - Property specification: A component will require various kinds of
        access to different properties (i.e. arrays representing them). This
        requirement can be specified in the following manner:
            
            - specify for each entity type, arrays that will be required for
              every particle.
            - specify for each entity type, property that will required for the
              whole entity.
            - specify for each entity type, property that will be required for
              every particle, but to be written into by only this component.
            - specify for each entity type, flags that will be required for
              every particle - also need to specify default value for flag. A
              flag will usually be specific to this component, or components
              that work as helpers to this component. Thus they are typically
              'private' properties to this component.
            - specify properties that need to be considered as 'output' of this
              component.

    """
    # list of information keys used by this class.
    PARTICLE_PROPERTIES_READ = 'PARTICLE_PROPERTIES_READ'
    PARTICLE_PROPERTIES_WRITE = 'PARTICLE_PROPERTIES_WRITE'
    PARTICLE_PROPERTIES_PRIVATE = 'PARTICLE_PROPERTIES_PRIVATE'
    PARTICLE_FLAGS = 'PARTICLE_FLAGS'

    ENTITY_PROPERTIES = 'ENTITY_PROPERTIES'
    OUTPUT_PROPERTIES = 'OUTPUT_PROPERTIES'

    INPUT_TYPES = 'INPUT_TYPES'
    ENTITY_NAMES = 'ENTITY_NAMES'

    def __cinit__(self, str name='', ComponentManager cm = None, *args, **kwargs):
        """
        Constructor.
        """
        self.name = name
        self.cm = cm

        self.information.set_dict(SolverComponent.PARTICLE_PROPERTIES_READ, {})
        self.information.set_dict(SolverComponent.PARTICLE_PROPERTIES_WRITE, {})
        self.information.set_dict(SolverComponent.PARTICLE_PROPERTIES_PRIVATE,
                                  {})
        self.information.set_dict(SolverComponent.PARTICLE_FLAGS, {})
        self.information.set_dict(SolverComponent.ENTITY_PROPERTIES, {})
        self.information.set_dict(SolverComponent.OUTPUT_PROPERTIES, {})
        self.information.set_dict(SolverComponent.INPUT_TYPES, {})
        self.information.set_dict(SolverComponent.ENTITY_NAMES, {})
        
    cpdef bint filter_entity(self, EntityBase entity):
        """
        Returns true if this entity fails any input requirement checks.

        **Algorithm**::

            if no INPUT_TYPES is specified or if this entity's type is accepted
                input type
                if there is not named entity requirement for entities of this
                type or if this entities name appears in the required names
                    return False
                return True

            return True        
          
        """
        cdef dict input_types = self.information.get_dict(
            SolverComponent.INPUT_TYPES)
        cdef dict entity_names = self.information.get_dict(
            SolverComponent.ENTITY_NAMES)

        cdef list req_names
        cdef bint type_accepted = False
        
        if len(input_types.keys()) > 0:
            for type in input_types.keys():
                if entity.is_a(type):
                    type_accepted = True
                    break
            # entity does not pass any type requirement
            # hence entity should be filtered.
            if not type_accepted:
                return True
        
        # entity has passed input type requirements or no input type
        # requirements were there for this component, hence further checks.

        # check if there is any named requirements for this entity's type
        req_names = entity_names.get(entity.type)

        if req_names is None:
            # all test passes, entity should not be filtered
            return False
            
        # check if this entity's name is there in list
        if entity.name in req_names:
            return False

        return True

    cpdef add_entity(self, EntityBase entity):
        """
        Add the given entity to the input if it passes all tests. This will be
        reimplemented in the derived classes.
        """
        raise NotImplementedError, 'SolverComponent::add_entity'

    cdef int compute(self) except -1:
        """
        Function where the core logic of the component is implemented. Implement
        this in your derived classes.

        """
        raise NotImplementedError, 'SovlerComponent::compute'

    cpdef int py_compute(self) except -1:
        """
        Python wrapper for the compute function.
        """
        return self.compute()

    cpdef int setup_component(self) except -1:
        """
        Setup internals of the component from the component info.
        """
        raise NotImplementedError, 'SolverComponent::setup_component'

################################################################################
# `UserDefinedComponent` class.
################################################################################
cdef class UserDefinedComponent(SolverComponent):
    """
    """
    def __cinit__(self, str name='', ComponentManager cm=None, *args, **kwargs):
        """
        Constructor.
        """
        pass

    cdef int compute(self) except -1:
        """
        Just calls the py_compute method.
        """
        return self.py_compute()

    cpdef int py_compute(self) except -1:
        """
        Function where the core logic of a user defined component is to be
        implemented. 
        """
        raise NotImplementedError, 'UserDefinedComponent::py_compute'

################################################################################
# `ComponentManager` class.
################################################################################
cdef class ComponentManager(Base):
    """
    Class to manage components.

    **Notes**
    
        - property_component_map - for every property, a list of dictionaries
          will be maintained. Each dictionary will have 2 keys - name and
          access. The name key will contain the name of the component requiring
          this component. The access key will have a dictionary, which will have
          one key for each entity type this property is required. For each
          entity, the access type of this component will be maintained.

        - particle_properties - A dictionary keyed on entity type, with a list
          for each key. Each list will contain one dictionary for each property
          required. This dictionary will have the following keys: name,
          data_type, default.
        
        - entity_properties - A dictionary keyed on entity type, with one
          property name for each key. All properties here will be double.

    """
    PARTICLE_PROPERTIES = 'PARTICLE_PROPERTIES'
    ENTITY_PROPERTIES = 'ENTITY_PROPERTIES'

    PROPERTY_COMPONENT_MAP = 'PROPERTY_COMPONENT_MAP'

    def __cinit__(self, *args, **kwargs):
        """
        Constructor.
        """
        self.component_dict = {}
        
        self.information.set_dict(SolverComponent.PARTICLE_PROPERTIES_READ, {})
        self.information.set_dict(SolverComponent.PARTICLE_PROPERTIES_WRITE, {})
        self.information.set_dict(SolverComponent.PARTICLE_PROPERTIES_PRIVATE,
                                  {})
        
        self.information.set_dict(ComponentManager.ENTITY_PROPERTIES, {})
        self.information.set_dict(ComponentManager.PARTICLE_PROPERTIES, {})
        
        # stores the names of components requiring a particular property.
        self.information.set_dict(
            ComponentManager.PROPERTY_COMPONENT_MAP, {})

    cpdef add_input(self, EntityBase entity):
        """
        Adds this entity to all entities in the component manager that need to
        be notified about new inputs.
        """
        cdef SolverComponent c

        for val in self.component_dict:
            if val['notify'] == True:
                c = val['component']
                c.add_entity(c)

    cpdef SolverComponent get_component(self, str comp_name):
        """
        Get the named component.
        """
        if self.component_dict.has_key(comp_name):
            return <SolverComponent>self.component_dict['comp_name']['component']
        else:
            logger.error('%s : no such component'%(comp_name))
            return None

    cpdef add_component(self, SolverComponent c, bint notify=False):
        """
        Add a new component to be managed.

        **Parameters**
        
            - c - the component to be added to the component manager.
            - notify - indicates if the component should be notified if
              some new entity is added as input.

        """
        # first check if the component already exists, in which case we won't
        # add it. Also two components of same name are not allowed.
        cdef dict comp_details = self.component_dict.get(c.name)
        if comp_details is not None:
            if comp_details['component'] is not c:
                raise ValueError, 'Two components with same name not allowed'
            return
        else:
            # add this component.
            if self.validate_property_requirements(c):
                self.component_dict[c.name] = {'component':c, 'notify':notify}
            else:
                logger.warn('Component %s not added'%(c.name))

    cpdef remove_component(self, str comp_name):
        """
        Remove a given component, and also property requirements of it, in case
        no other component requires the same.

        """
        pass

    cpdef bint validate_property_requirements(self, SolverComponent c) except *:
        """
        Returns true and updates property requirements if the requirements are
        valid (no conflicts with other components). Returns false otherwise, the
        property requirements are not updated.

        **Algorithm**::

               - private properties of this component should not appear in the
                 write/private properties of any other component.
         
               - write properties of this component should not be private
                 properties in any other component.

               - flags of this component should not be write/flags in any other
                 component, except if the component are related somehow.

               - write/flags of this component should not be flags of already
                 existing components.

        """
        cdef dict p_props, r_props, w_props, flags
        cdef dict entity, particle

        cdef TypedDict c_info = c.information
        
        p_props = c_info.get_dict(SolverComponent.PARTICLE_PROPERTIES_PRIVATE)
        w_props = c_info.get_dict(SolverComponent.PARTICLE_PROPERTIES_WRITE)
        r_props = c_info.get_dict(SolverComponent.PARTICLE_PROPERTIES_READ)
        flags = c_info.get_dict(SolverComponent.PARTICLE_FLAGS)
        
        # check the validity of private properties.
        for etype in p_props:
            p_list = p_props[etype]
            for p in p_list:
                if not self._check_property(p, 'private', etype):
                    logger.error('Failed to add component %s'%(c.name))
                    return False
        for etype in w_props:
            p_list = w_props[etype]
            for p in p_list:
                if not self._check_property(p, 'write', etype):
                    logger.error('Failed to add component %s'%(c.name))
                    return False
        for etype in flags:
            flag_list = flags[etype]
            for f in flag_list:
                if not self._check_property(f, 'write', etype):
                    logger.error('Failed to add component %s'%(c.name))
                    return False
                
        # all checks done, this property can be added to the list of entity
        # properties and other tables.

        # add entity properties to entity_properties
        entity_props = self.information.get_dict(
            ComponentManager.ENTITY_PROPERTIES)

        e_props = c.information.get_dict(
            c.ENTITY_PROPERTIES)
        
        for etype in e_props:
            p_list = e_props[etype]

            e_props1 = entity_props.get(etype)

            if e_props1 is None:
                e_props1 = {}
                entity_props[etype] = e_props1
            
            for p in p_list:
                p_name = p['name']
                p_default = p['default']

                p1 = e_props1.get(p_name)
                if p1 is None:
                    p1 = {'default':p_default}
                    e_props1[p_name] = p1
                else:
                    if p1['default'] is None:
                        p1['default'] = p_default
                    else:
                        if p_default != p1['default']:
                            msg = 'Different default values for'
                            msg += ' %s'%(p_name)
                            logger.warn(msg)
                            logger.warn('Using new value of %s'%p_default)
                            p1['default'] = p_default                
        
        # add the particle properites to particle_properties
        particle_props = self.information.get_dict(
            ComponentManager.PARTICLE_PROPERTIES)
        
        # add the private particle properties.
        for etype in p_props:
            # get the properties for this component
            p_list = p_props[etype]
            for p in p_list:
                self._update_property_component_map(p['name'], c.name, 
                                                    etype, 'private')
                self._add_particle_property(p, etype)

        for etype in r_props:
            p_list = r_props[etype]
            # r props will just be a list of property names
            # create a dict for each and add.
            for p in p_list:
                d = {'name':p, 'default':None}
                self._update_property_component_map(p, c.name, 
                                                    etype, 'read')
                self._add_particle_property(d, etype)

        for etype in w_props:
            p_list = w_props[etype]
            for p in p_list:
                self._update_property_component_map(p['name'], c.name, 
                                                    etype, 'write')
                self._add_particle_property(p, etype)

        for etype in flags:
            flag_list = flags[etype]
            for f in flag_list:
                self._update_property_component_map(p['name'], c.name, 
                                                    etype, 'write')
                self._add_particle_property(f, etype, 'int')

        return True

    cpdef _update_property_component_map(self, str prop, str comp_name, int etype, str
                                         access_type):
        """
        """
        cdef dict pc_map = self.information.get_dict(
            self.PROPERTY_COMPONENT_MAP)

        # get the dictionary maintained for this property
        cdef dict comp_dict = pc_map.get(prop)

        if comp_dict is None:
            comp_dict = {}
            pc_map[prop] = comp_dict

        # get the dictionary maintained for given component
        cdef dict comp = comp_dict.get(comp_name)
        
        if comp is None:
            comp = {}
            comp['name'] = comp_name
            comp['access'] = {}
            comp_dict[comp_name] = comp

        # get the dictionary maintained for this entity type.
        cdef str access = comp['access'].get(etype)

        if access is None:
            comp['access'][etype] = access_type
        else:
            if comp['access'][etype] != access_type:
                logger.warn('Conflicting access types for same property')
                msg = 'Make sure each property appears in exactly one class'
                logger.warn(msg)
        
    cpdef _add_particle_property(self, dict prop, int etype, str data_type='double'):
        """
        """
        cdef dict particle_props = self.information.get_dict(
            ComponentManager.PARTICLE_PROPERTIES)

        cdef dict e_type_props = particle_props.get(etype)

        # no property for this entity type yet.
        if e_type_props is None:
            particle_props[etype] = {}
            e_type_props = particle_props[etype]

        pname = prop['name']
        entry = e_type_props.get(pname)
        
        if entry is None:
            d = {}
            d['data_type'] = data_type
            d['default'] = prop['default']
            e_type_props[pname] = d
        else:
            if prop['default'] is not None:
                if entry['default'] is None:
                    entry['default'] = prop['default']
                elif entry['default'] != prop['default']:
                    msg = 'Updating default value to'
                    msg = '%f, for property %s'%(
                        prop['default'], pname)
                    logger.warn(msg)
                    entry['default'] = prop['default']                    
        
    cpdef bint _check_property(self, dict prop, str access_mode, int etype) except *:
        """
        Check if this property is safe.
        """
        cdef str prop_name, c, c_name, access_mode_1
        cdef dict pc_map, c_inf, access

        prop_name = prop['name']
        pc_map = self.information.get_dict(
            ComponentManager.PROPERTY_COMPONENT_MAP)

        # get the list of components needing this property.
        # this is stored a a dictionary, keyed on the components name, and
        # holding the additionary information.
        c_inf = pc_map.get(prop_name)

        # no component yet needs this property
        if c_inf is None:
            return True

        # checks
        for c in c_inf.keys():
            c_name = c
            access = c_inf[c_name]['access']

            access_mode_1 = access.get(etype)

            # no component yet needs this property for this
            # entity type, with this access.
            if access_mode == 'private':
                if (access_mode_1 == 'private' or
                    access_mode_1 == 'write'):
                    # we cannot allow private of the input component to be a write
                    # or a private property of an already existing component.
                    msg = 'Property %s is %s in %s'%(prop_name, access_mode_1,
                                                     c_name)
                    logger.error(msg)
                    return False
                if access_mode == 'write':
                    if (access_mode_1 == 'private'):
                        # we cannot allow any component to write to a private
                        # property of another component.
                        msg = 'Property %s is %s in %s'%(prop_name, access_mode_1,
                                                         c_name)
                        logger.error(msg)
                        return False
        
        return True