"""
Base class for all solver components.
"""

# logger imports
import logging
logger = logging.getLogger()
cimport numpy
import numpy

# package imports
import random

# local imports
from pysph.base.kernels cimport KernelBase
from pysph.base.nnps cimport NNPSManager
from pysph.base.cell cimport CellManager

#from pysph.solver.integrator_base import Integrator
import pysph.solver.integrator_base as integrator_base
from pysph.solver.entity_base cimport EntityBase
from pysph.solver.time_step cimport TimeStep
from pysph.solver.speed_of_sound cimport SpeedOfSound
from pysph.solver.nnps_updater cimport NNPSUpdater
from pysph.solver.timing import Timer

# FIXME
# 1. The compute function finally implemented by the USER should not require
# him/her to call the setup_component function. It should be done
# internally. This could be done by implementing the base class compute to call
# setup_component and then call a compute_def function - which will be written
# by the user.
#
# 2. Default values of properties should/need not be specified by the
# component. It may have to be specified somewhere else.
################################################################################
# `SolverComponent` class.
################################################################################
cdef class SolverComponent:
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
              'write' properties to this component.
            - specify properties that need to be considered as 'output' of this
              component.
      
    """
    # list of information keys used by this class.
    identifier = 'base_component'
    category = 'base'

    def __cinit__(self, str name='', SolverBase solver = None, ComponentManager
                  component_manager=None, list entity_list=[], *args, **kwargs):
        """
        Constructor.
        """
        self.name = name
        self.solver = solver

        if solver is not None:
            self.cm = solver.component_manager
        else:
            self.cm = component_manager

        self.setup_done = False
        self.accept_input_entities = True
        self.entity_list = []
        self.entity_list[:] = entity_list

        self.particle_props_read = {}
        self.particle_props_write = {}
        self.particle_props_private = {}
        self.particle_flags = {}
        self.entity_props = {}
        self.input_types = set()
        self.entity_names = {}
    
    def __init__(self,  *args, **kwargs):
        """
        Python constructor.
        """
        pass

    cpdef bint filter_entity(self, EntityBase entity):
        """
        Returns true if this entity fails any input requirement checks.

        **Algorithm**::

            if accept_input_entities is set to False, no entities will be
            accepted.

            if no input_types is specified or if this entity's type is accepted
                input type
                if there is not named entity requirement for entities of this
                type or if this entities name appears in the required names
                    return False
                return True

            return True        
          
        """
        # if the component has been flagged to stop accepting input entites
        # immediately filter out entity.
        if self.accept_input_entities == False:
            return True

        cdef set input_types = self.input_types
        cdef dict entity_names = self.entity_names

        cdef list req_names
        cdef bint type_accepted = False
        
        if len(input_types) > 0:
            for type in input_types:
                if entity.is_a(type):
                    type_accepted = True
                    break
            # entity does not pass any type requirement
            # hence entity should be filtered.
            if not type_accepted:
                return True
        
        # entity has passed input type requirements or no input type
        # requirements were there for this component, hence further checks.
        return False

    cpdef add_entity_name(self, str name):
        """
        Add name of an entity that can be accepted as input.
        """
        raise NotImplementedError, 'SolverComponent::add_entity_name'
    
    cpdef remove_entity_name(self, str name):
        """
        Remove name of an entity that was added.
        """
        raise NotImplementedError, 'SolverComponent::remove_entity_name'
        
    cpdef set_entity_names(self, list entity_names):
        """
        Sets the entity names list to the given list.
        """
        raise NotImplementedError, 'SolverComponent::set_entity_names'
        
    cpdef add_input_entity_type(self, type etype):
        """
        Adds an entity type that will be accepted by this component.
        """
        self.input_types.add(etype)

    cpdef remove_input_entity_type(self, type etype):
        """
        Removes a particular entity type that was added.
        """
        self.input_types.remove(etype)

    cpdef set_input_entity_types(self, list type_list):
        """
        Sets the accepted entity types from the given list.
        """
        self.input_types.clear()
        self.input_types.update(set(type_list))

    cpdef add_read_prop_requirement(self, type e_type, list prop_list):
        """
        Adds a property requirement for the given type.

        prop_list is a list of strings, one for each property.
        """
        cdef dict rp = self.particle_props_read
        cdef set t_rp = rp.get(e_type)

        if t_rp is None:
            t_rp = set([])
            rp[e_type] = t_rp

        t_rp.update(set(prop_list))

    cpdef add_write_prop_requirement(self, type e_type, str prop_name, double
                                     default_value=0.0):
        """
        Adds a property requirement for the given type.
        """
        cdef dict wp = self.particle_props_write
        cdef list t_wp = wp.get(e_type)
        
        if t_wp is None:
            t_wp = []
            wp[e_type] = t_wp

        e = {'name':prop_name, 'default':default_value}

        t_wp.append(e)

    cpdef add_private_prop_requirement(self, type e_type, str prop_name, double
                                       default_value=0.0):
        """
        Adds a property requirement for the given type.
        """
        cdef dict pp = self.particle_props_private
        cdef list t_pp = pp.get(e_type)
        
        if t_pp is None:
            t_pp = []
            pp[e_type] = t_pp

        e = {'name':prop_name, 'default':default_value}

        t_pp.append(e)

    cpdef add_flag_requirement(self, type e_type, str flag_name, int
                               default_value=0):
        """
        Adds a flag property requirement for the given type.
        """
        cdef dict f = self.particle_flags
        cdef list t_f = f.get(e_type)
        
        if t_f is None:
            t_f = []
            f[e_type] = t_f

        e = {'name':flag_name, 'default':default_value}

        t_f.append(e)
        
    cpdef add_entity_prop_requirement(self, type e_type, str prop_name, double
                                      default_value=0.0):
        """
        Add a property requirement of the entity.
        """
        cdef dict ep = self.entity_props
        cdef list t_ep = ep.get(e_type)

        if t_ep is None:
            t_ep = []
            ep[e_type] = t_ep

        e = {'name':prop_name, 'default':default_value}
        
        t_ep.append(e)

    cpdef add_entity(self, EntityBase entity):
        """
        Add the given entity to the entity_list if filter_entity does not filter
        out this entity based on the input requirements.

        Derived classes may reimplement this function as needed.

        """
        if self.filter_entity(entity) == False:
            if self.entity_list.count(entity) == 0:
                self.entity_list.append(entity)
                self.setup_done = False

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

    cpdef int update_property_requirements(self) except -1:
        """
        Make up to date any property/array requirements of this component.


        Most components will be able to specify their property requirements
        during construction time. Thus this function need NOT be implemented in
        most cases. In some components however, property requirements become
        available only when the component is being setup (using various public
        function provided by the component) We are NOT referring to the final
        setup of the component before execution.
        
        This function should be written such that, once the component has been
        setup, this function is able to dedude the property requirements from
        the components internal information (specific to the component).

        This will be used by the component manager before it queries a component
        for its property requirement. Once a component has been setup, this
        function's operation should be idempotent - calling the function
        repeatedly should not change the property requirements.

        This function should be only called AFTER the component has been setup
        as required. The component manager will call this function inside the
        add_component function. Property requirements should not change after
        the component has been added to the component manager.

        """
        return 0

################################################################################
# `UserDefinedComponent` class.
################################################################################
cdef class UserDefinedComponent(SolverComponent):
    """
    """
    category = 'base'
    identifier = 'ud_component_base'
    def __cinit__(self, str name='', SolverBase solver=None, ComponentManager
                  component_manager=None, list entity_list=[], *args, **kwargs):
        """
        Constructor.
        """
        pass

    def __init__(self, *args, **kwargs):
        """
        Python constructor.
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
cdef class ComponentManager:
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
          type, default.
        
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

        self.particle_props = {} 
        self.entity_props = {}
        self.property_component_map = {}
        self.particle_props_read = {}
        self.particle_props_write = {}
        self.particle_props_private = {}
        
    cpdef get_entity_properties(self, type e_type):
        """
        Get the entity property requirements of all components.
        """
        return self.entity_props[e_type]

    cpdef get_particle_properties(self, type e_type):
        """
        Get the particle property requirements of all components.
        """
        return self.particle_props[e_type]

    cpdef add_input(self, EntityBase entity):
        """
        Adds this entity to all entities in the component manager that need to
        be notified about new inputs.
        """
        cdef SolverComponent c

        for val in self.component_dict.values():
            if val['notify'] == True:
                c = val['component']
                c.add_entity(entity)

    cpdef SolverComponent get_component(self, str comp_name):
        """
        Get the named component.
        """
        if self.component_dict.has_key(comp_name):
            return <SolverComponent>self.component_dict[comp_name]['component']
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

        **Notes**

            - Property requirements of the component should not change after it
              has been added to the component manager. The component manager
              will not be able to consistently report the property requirements
              if that is done.

        """
        # first check if the component already exists, in which case we won't
        # add it. Also two components of same name are not allowed.
        cdef dict comp_details = self.component_dict.get(c.name)
        if comp_details is not None:
            if comp_details['component'] is not c:
                raise ValueError, 'Two components with same name not allowed'
            return
        else:
            # update the properties component requirements.
            c.update_property_requirements()

            # add this component.
            if self.validate_property_requirements(c):
                self.component_dict[c.name] = {'component':c, 'notify':notify}
            else:
                logger.warn('Component %s not added'%(c.name))
                raise ValueError, 'Component %s not added'%(c.name)

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

        p_props = c.particle_props_private
        w_props = c.particle_props_write
        r_props = c.particle_props_read
        flags = c.particle_flags
        
        # check the validity of private properties.
        for etype in p_props:
            p_list = p_props[etype]
            for p in p_list:
                if not self._check_property(c, p, 'private', etype):
                    logger.error('Failed to add component %s'%(c.name))
                    return False
        for etype in w_props:
            p_list = w_props[etype]
            for p in p_list:
                if not self._check_property(c, p, 'write', etype):
                    logger.error('Failed to add component %s'%(c.name))
                    return False
        for etype in flags:
            flag_list = flags[etype]
            for f in flag_list:
                if not self._check_property(c, f, 'write', etype):
                    logger.error('Failed to add component %s'%(c.name))
                    return False
                
        # all checks done, this property can be added to the list of entity
        # properties and other tables.

        # add entity properties to entity_properties
        entity_props = self.entity_props
        e_props = c.entity_props
        
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
        
        # add the particle properties to particle_properties
        particle_props = self.particle_props
        
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

    cpdef _update_property_component_map(self, str prop, str comp_name, type
                                         etype, str access_type):
        """
        """
        pc_map = self.property_component_map

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
        
    cpdef _add_particle_property(self, dict prop, type etype, str data_type='double'):
        """
        """
        cdef dict particle_props = self.particle_props
        cdef dict e_type_props = particle_props.get(etype)

        # no property for this entity type yet.
        if e_type_props is None:
            particle_props[etype] = {}
            e_type_props = particle_props[etype]

        pname = prop['name']
        entry = e_type_props.get(pname)
        
        if entry is None:
            d = {}
            d['type'] = data_type
            d['default'] = prop['default']
            d['name'] = pname
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
        
    cpdef bint _check_property(self, SolverComponent comp, dict prop, str access_mode, type etype) except *:
        """
        Check if this property is safe.
        """
        cdef str prop_name, c, c_name, access_mode_1
        cdef dict pc_map, c_inf, access
        cdef SolverComponent existing_comp

        prop_name = prop['name']
        pc_map = self.property_component_map

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
            existing_comp = self.get_component(c_name)
            
            if type(existing_comp) == type(comp):
                continue
            
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

    cpdef setup_entity(self, EntityBase entity):
        """
        Sets up required properties for the given entity.

        Call this on any entity AFTER all required components have been added to
        the component manager.
        """
        ep = self.entity_props
        pp = self.particle_props

        # update the entity properties first
        for e_type in ep.keys():
            if entity.is_a(e_type):
                e_props = ep[e_type]
                for prop_name in e_props:
                    if entity.properties.has_key(prop_name):
                        continue
                    else:
                        entity.properties[prop_name] = e_props[prop_name][
                            'default']
        
        # update the particle properties
        parray = entity.get_particle_array()
        if parray is None:
            return
        
        for e_type in pp.keys():
            if entity.is_a(e_type):
                p_props = pp[e_type]
                for prop_name in p_props:
                    prop_inf = p_props[prop_name]
                    parray.add_property(prop_inf)
        
###############################################################################
# `SolverBase` class.
###############################################################################
cdef class SolverBase:
    """
    """
    def __cinit__(self,
                  ComponentManager component_manager=None,
                  CellManager cell_manager=None,
                  NNPSManager nnps_manager=None,
                  KernelBase kernel=None,
                  object integrator=None,
                  double time_step=0.0,
                  double total_simulation_time=0.0,
                  bint enable_timing=False,
                  str timing_output_file='',
                  *args, **kwargs):
        """
        Cython constructor.
        """
        pass

    def __init__(self, 
                 ComponentManager component_manager=None,
                 CellManager cell_manager=None,
                 NNPSManager nnps_manager=None,
                 KernelBase kernel=None,
                 object integrator=None,
                 double time_step=0.0,
                 double total_simulation_time=0.0,
                 bint enable_timing=False,
                 str timing_output_file='',
                 *args, **kwargs
                  ):
        """
        Python constructor.
        """        
        if component_manager is None:
            self.component_manager = ComponentManager()
        else:
            self.component_manager = component_manager

        if cell_manager is None:
            self.cell_manager = CellManager(initialize=False)
        else:
            logger.debug('Using cell manager : %s'%(cell_manager))
            self.cell_manager = cell_manager
        
        if nnps_manager is None:
            self.nnps_manager = NNPSManager(cell_manager=self.cell_manager)
        else:
            self.nnps_manager = nnps_manager

        # default kernel to be used by all components.
        self.kernel = kernel
        # the time step variable accessed throughout the solver.
        self.time_step = TimeStep(time_step)

        # total simulation time that has be completed.
        self.elapsed_time = 0.0
        # total time to run the simulation for.
        self.total_simulation_time = total_simulation_time
        # current iteration.
        self.current_iteration = 0

        self.entity_list=[]
        self.component_categories = {}

        # setup some standard component categories.
        self.component_categories['pre_integration'] = []
        self.component_categories['post_integration'] = []
        self.component_categories['pre_step'] = []
        self.component_categories['post_step'] = []
        self.component_categories['pre_iteration'] = []

        # add a nnps_updater component to the pre-step components.
        psc = self.component_categories['pre_step']
        nnps_updater = NNPSUpdater(solver=self)
        psc.append(nnps_updater)

        self.kernels_used = {}
        
        # the integrator.
        self.integrator = integrator
        if self.integrator is None:
            self.integrator = integrator_base.Integrator(name='integrator_default', solver=self)
        self.integrator.time_step = self.time_step

        # the timer
        self.enable_timing = enable_timing
        self.timing_output_file = timing_output_file
        self.timer = Timer(output_file_name=self.timing_output_file)

    cpdef add_entity(self, EntityBase entity):
        """
        Add a new entity to be included in the simulation.
        """
        if self.entity_list.count(entity) == 0:
            self.entity_list.append(entity)

    cpdef register_kernel(self, KernelBase kernel):
        """
        Adds the given kernel into the dict kernels_used.
        """
        if kernel is None:
            return

        if not self.kernels_used.has_key(type(kernel)):
            self.kernels_used[type(kernel)] = kernel
            
    cpdef solve(self):
        """
        Run the solver.
        """
        cdef SolverComponent c

        self._setup_solver()

        current_time = 0.0

        # execute any pre-iterations components before the iterations begin.
        pre_iteration_components = self.component_categories['pre_iteration']
        for cm in pre_iteration_components:
            c = <SolverComponent>cm
            c.compute()

        while current_time < self.total_simulation_time:

            logger.info('Iteration %d start \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ '%(
                    self.current_iteration))

            if self.enable_timing:
                self.timer.start()

            self.integrator.integrate()

            logger.debug('Execute list : %s'%(self.integrator.execute_list))

            if self.enable_timing:
                self.timer.finish()

            logger.info('Iteration %d done %f \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ '%(
                    self.current_iteration, self.elapsed_time))

            current_time += self.time_step.value
            self.elapsed_time = current_time    
            self.current_iteration += 1

    cpdef next_iteration(self):
        """
        Advance simulation by one time step.
        """
        cdef double t = self.elapsed_time
        
        if t < self.total_simulation_time:
            if abs(t - self.total_simulation_time) < 1e-12:
                pass
            else:
                self.integrator.integrate()
                self.elapsed_time += self.time_step.value
                self.current_iteration += 1

    cpdef _setup_solver(self):
        """
        Function to perform solver setup.
        """
        logger.info('Setting up integrator ...')
        self._setup_integrator()

        logger.info('Setting up component manager ...')
        self._setup_component_manager()

        logger.info('Setting up entities ...')
        self._setup_entities()

        logger.info('Setting up nnps ...')
        self._setup_nnps()

        logger.info('Setting up component inputs ...')
        self._setup_components_input()

    cpdef _setup_component_manager(self):
        """
        Add all the components (in component_categories) into the component
        manager.

        - If no component name is given for the component, generate a name
        before adding the component to the component manager.

        - Make sure the "solver" attribute of each component is set to "self".
        
        """
        for comp_category in self.component_categories.keys():
            comp_list = self.component_categories[comp_category]
            i=0
            for c in comp_list:
                c.solver = self
                self._component_name_check(c)
                logger.info('Adding component %s to component manager'%(c.name))
                self.component_manager.add_component(c, notify=True)
                i += 1
                
        # add the integrator to the component manager
        self.integrator.solver = self
        if self.integrator.name == '':
            self.integrator.name = 'integrator_default'
        self.component_manager.add_component(self.integrator, notify=True)

    cpdef _setup_entities(self):
        """
        For each entity in the entity_dict, set the required properties in the
        entities using the component manager.

        If any entity has not been named, raise an error here.
        """
        for e in self.entity_list:
            if e.name == '':
                msg = 'Name not set for entity %s'%(e)
                logger.error(msg)
                raise ValueError, msg

            self.component_manager.setup_entity(e)
            
    cpdef _setup_nnps(self):
        """
        Setup the nnps and the cell manager.

        Operations involved.
        
            - Add all possible particle arrays to the cell manager to be binned.
            - Compute the cell size to be used in the cell manager.
            
        """
        for e in self.entity_list:
            e.add_arrays_to_cell_manager(self.cell_manager)

        min_cell_size, max_cell_size = self._compute_cell_sizes()

        if min_cell_size != -1 and max_cell_size != -1:
            self.cell_manager.min_cell_size = min_cell_size
            self.cell_manager.max_cell_size = max_cell_size

        # initialize the cell manager.
        self.cell_manager.initialize()

    def _component_name_check(self, SolverComponent c):
        """
        """
        if c.name == '':
            c.name = self._gen_random_name_for_component(c)
            logger.warn('Using name %s for component %s'%(
                    c.name, c))
    
    def _gen_random_name_for_component(self, SolverComponent c):
        """
        """
        r = int(random.random()*100)
        return c.category+'_'+c.identifier+'_'+str(r)
        
    cpdef _setup_components_input(self):
        """
        Add each entity in the entity_list to all components.
        
        The entities will be appropriately filtered based on how the component
        is setup.
        """
        for e in self.entity_list:
            self.component_manager.add_input(e)

    cpdef _compute_cell_sizes(self):
        """
        Find the minimum 'h' value from all particle arrays of all entities.
        Use twice the size as the cell size.

        This is very simplistic method to find the cell sizes, derived solvers
        may want to use something more sophisticated or probably set the cell
        sizes manually.
        """
        cdef double min_h = 100.0
        cdef bint size_computed = False
        for e in self.entity_list:
            parr = e.get_particle_array()
            if parr is None or parr.get_number_of_particles() == 0:
                continue
            h = parr.h
            if h is None:
                continue
            min_h1 = numpy.min(h)*2.0
            size_computed = True
            if min_h1 < min_h:
                min_h = min_h1

        if size_computed == False:
            logger.info('No particles found - using default cell sizes')
            return -1, -1

        logger.info('using cell size of %f'%(min_h))
        return min_h, min_h

    cpdef _setup_integrator(self):
        """
        Setup the integrator as required. 
        Some basic setup is done here if needed.

        The most important parts will be domain specific and be done in the
        derived solver classes.
        """
        logger.warn('No integrator setup in SolverBase')
        
        
