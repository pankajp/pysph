"""
Contains base classes for all integrators.
"""

# logger import
import logging
logger = logging.getLogger()

# numpy imports
cimport numpy

# local imports
from pysph.base.carray cimport DoubleArray
from pysph.base.particle_array cimport ParticleArray
from pysph.solver.solver_component cimport SolverComponent, ComponentManager
from pysph.solver.entity_base cimport EntityBase
from pysph.solver.entity_types cimport EntityTypes

# import component factory
from pysph.solver.component_factory import ComponentFactory as cfac

################################################################################
# `TimeStep` class.
################################################################################
cdef class TimeStep(Base):
    """
    Class to hold the current timestep.

    Make this a separate class makes it easy to reference one copy of it at all
    places needing this value.

    """
    def __cinit__(self, double time_step=0.0):
        """
        Constructor.
        """
        self.time_step = time_step    

################################################################################
# `ODEStepper` class.
################################################################################
cdef class ODEStepper(SolverComponent):
    """
    Class to step a given property by a given time step.

    This class implements a simple euler step. The values of the next time step
    are stored in arrays, whose names are got by suffixing _next to the original
    array name.

    """
    category='ode_stepper'
    identifier = 'base'
    def __cinit__(self, str name='', ComponentManager cm=None, list
                  entity_list=[], str prop_name='', list integrands=[], list
                  integrals=[], TimeStep time_step=None, *args, **kwargs):
        """
        Constructor.

        **Params**
        
             - entity_list - list of entities (not names).
             - prop_name - name of the property being stepped. This parameters
               is required as some steppers may have specific setup tasks to do
               for particular properties.
             - integrand_names - list of names of integrands.
             - integral_names - list of names of integrals.

        """
        self.entity_list = []
        self.prop_name = ''
        self.integrand_names = []
        self.integral_names = []
        self.next_step_names = []

        self.time_step = time_step

        self.entity_list[:] = entity_list

        self.set_properties(prop_name, integrands, integrals)

    cpdef set_properties(self, str prop_name, list integrands, list integrals):
        """
        Sets the properties that are to be stepped.
        """
        self.prop_name = prop_name
        self.integral_names[:] = integrals
        self.integrand_names[:] = integrands
        self.setup_done = False

    cpdef int setup_component(self) except -1:
        """
        Sets up the names of the arrays to be used for storing the values of the
        next step.
        """
        cdef int i, num_props, num_entities
        cdef str arr_name
        cdef EntityBase e
        cdef ParticleArray parr
        cdef list to_remove = []

        if self.setup_done == False:

            # make sure timestep is setup properly
            if self.time_step is None:
                raise ValueError, 'time_step not set'

            self.next_step_names[:] = []
        
            num_props = len(self.integral_names)
        
            for i from 0 <= i < num_props:
                arr_name = self.integral_names[i]
                self.next_step_names.append(
                    arr_name + '_next')

            # now make sure that all the entities have the _next property.
            num_entities = len(self.entity_list)
            for i from 0 <= i < num_entities:
                e = self.entity_list[i]
                parr = e.get_particle_array()
                if parr is not None:
                    # make sure all the _next arrays are present, if not add
                    # them.
                    for j from 0 <= j < num_props:
                        arr_name = self.next_step_names[j]
                        if not parr.properties.has_key(arr_name):
                            parr.add_property({'name':arr_name})
                else:
                    to_remove.append(e)

            # remove entities that did not provide a particle array.
            for e in to_remove:
                self.entity_list.remove(e)
            
            # mark component setup as done.
            self.setup_done = True
        
    cdef int compute(self) except -1:
        """
        Performs simple euler integration by the time_step, for the said arrays
        for each entity.

        Each entity 'MUST' have a particle array representation. Otherwise
        stepping won't be done currently.

        """
        cdef EntityBase e
        cdef int i, num_entities
        cdef int num_props, p, j
        cpdef ParticleArray parr
        cdef numpy.ndarray an, bn, an1
        cdef DoubleArray _an, _bn, _an1
       
        # make sure the component has been setup
        self.setup_component()
        
        num_entities = len(self.entity_list)
        num_props = len(self.integrand_names)

        for i from 0 <= i < num_entities:
            e = self.entity_list[i]
            
            parr = e.get_particle_array()
    
            if parr is None:
                logger.warn('no particle array for %s'%(e.name))
                continue
            
            for j from 0 <= j < num_props:
                _an = parr.get_carray(self.integral_names[j])
                _bn = parr.get_carray(self.integrand_names[j])
                _an1 = parr.get_carray(self.next_step_names[j])

                an = _an.get_npy_array()
                bn = _bn.get_npy_array()
                an1 = _an1.get_npy_array()

                an1[:] = an + bn*self.time_step.time_step                

    cpdef add_entity(self, EntityBase entity):
        """
        Add an entity whose properties are to be integrated.
        """
        if not self.filter_entity(entity):
            self.entity_list.append(entity)
            self.setup_done = False

################################################################################
# `PyODEStepper` class.
################################################################################
cdef class PyODEStepper(ODEStepper):
    """
    Simple class to implement some ODEStepper in python.
    """
    category='ode_stepper'
    identifier='py_base'
    def __cinit__(self, name='', ComponentManager cm=None, entity_list=[], 
                  integrand_arrays=[], integral_arrays=[], *args, **kwargs):
        """
        Constructor.
        """
        pass

    cpdef int py_compute(self) except -1:
        """
        """
        raise NotImplementedError, 'PyODEStepper::py_compute'

    cdef int compute(self) except -1:
        """
        """
        self.py_compute()
################################################################################
# `Integrator` class.
################################################################################
cdef class Integrator(SolverComponent):
    """
    Base class for all integrators. Integrates a set of given properties.

    **Notes**

    **PRE_INTEGRATION_COMPONENTS**
        A list of components that should be executed before integration of any
        property begins.
    
    **INTEGRATION_PROPERTIES**
        For every entity type that can be integrated by this integrator, there
        will be an entry in this dictionary. Each entry will be a dictionary
        keyed on the property to be integrated, where the entry will be again a
        dictionary with the following keys:

            - integrand - a list of arrays making up the integrand.
            - integral  - a list of arrays (of same size as integrand), making
              up the integral.
            - pre_step - a list of components to be executed before
              stepping this property.
            - post_step - a list of components to be executed after stepping
              this property.
            - steppers - a dictionary, specifying the stepper to be used for
              each kind of entity, or a default stepper to be used for any
              entity while stepping the said property.
            - entity_types - a list of types of entities that will be accepted
              for stepping this particular property.

    **INTEGRATION_ORDER**
        A list of property names in the order in which they have to be stepped.

    **DEFAULT_STEPPERS**
        For each entity type, the kind of stepper to use. These stepper are kind
        of fall-back stepper / default stepper for each entity type, in case not
        other stepper is available. Usually integrators may decided to use a
        particular kind of stepper for a particular property. When no such
        information is provided, then default steppers from this will be
        used. By default the 'steppers' information should provide a 'default'
        stepper'. This will be used when no other stepper specification is
        found. In addition to this, an integrator class may decide to provide
        default steppers for each entity type. 

        Derived classes may use this for more complex specification of the
        steppers to be used.

    """
    # list of information keys provided by this class.
    INTEGRATION_PROPERTIES = 'INTEGRATION_PROPERTIES'
    PRE_INTEGRATION_COMPONENTS = 'PRE_INTEGRATION_COMPONENTS'
    DEFAULT_STEPPERS = 'DEFAULT_STEPPERS'
    INTEGRATION_ORDER = 'INTEGRATION_ORDER'
    
    identifier = 'euler'
    category = 'integrator'
    def __cinit__(self, str name='', ComponentManager cm=None, int dimension=3,
                  *args, **kwargs):
        """
        Constructor.
        """
        self.entity_list = set()
        self.execute_list = []

        self.information.set_dict(self.INTEGRATION_PROPERTIES, {})
        self.information.set_list(self.PRE_INTEGRATION_COMPONENTS, [])
        self.information.set_list(self.INTEGRATION_ORDER, [])
        self.information.set_dict(self.DEFAULT_STEPPERS, {})

        # setup the velocity and position properties according to dimension
        self.set_dimension(dimension)
        
        # setup the various steppers.
        self.setup_defualt_steppers()

    def setup_defualt_steppers(self):
        """
        Sets up the different kinds of steppers required by the integrator.
        
        This integrator will use an euler stepper for integrating any proerpty
        of any entity type. Change it as necessary for derived classes.
        """
        cdef dict default_steppers = self.information.get_dict(
            self.DEFAULT_STEPPERS) 
        cdef str s        
        # we only add one default stepper to be used for all kinds of entities. 
        s = default_steppers.get('default')
        
        if s is not None:
            logger.warn('Default stepper %s already exists'%(s))
            logger.warn('Replacing with euler')
        
        default_steppers['default'] = 'euler'        
    
    cpdef set_dimension(self, int dimension):
        """
        Sets the dimension of the velocity and position vectors.
        """
        # sanity checks
        if dimension <= 0:
            logger.warn('Dimension of <0 specified')
            logger.warn('Not adding velocity and position properites')
            return
        if dimension > 3:
            dimension = 3
            logger.warn('Dimension > 3 specified, using 3')

        self.dimension = dimension

        vel_arrays = []
        accel_arrays = []
        pos_arrays = []

        # add components as necessary.
        vel_arrays.append('u')
        accel_arrays.append('ax')
        pos_arrays.append('x')

        if dimension > 1:
            vel_arrays.append('v')
            accel_arrays.append('ay')
            pos_arrays.append('y')
        if dimension > 2:
            vel_arrays.append('w')
            accel_arrays.append('az')
            pos_arrays.append('z')
            
        # now add the velocity and position properties to be stepped.
        self.add_property('velocity', accel_arrays, vel_arrays,
                          [EntityTypes.Entity_Base])
        self.add_property('position', vel_arrays, pos_arrays,
                          [EntityTypes.Entity_Base])

    cpdef add_entity(self, EntityBase entity):
        """
        Adds an entity whose properties are to be integrated.
        """
        if self.filter_entity(entity):
            self.entity_list.add(entity)
            self.setup_done = False

    cpdef add_entity_type(self, str prop_name, int entity_type):
        """
        Includes the given entity type for integration of a given property.
        """
        cdef dict ip = self.information.get_dict(self.INTEGRATION_PROPERTIES)
        cdef dict prop_info = ip.get(prop_name)
        
        if prop_info is None:
            logger.warn('No such property %s'%(prop_name))
            return

        entity_types = prop_info.get('entity_types')

        if entity_types is None:
            entity_types = []
            prop_info['entity_types'] = entity_types
        
        if entity_types.count(entity_type) == 0:
            entity_types.append(entity_type)

    cpdef remove_entity_type(self, str prop_name, int entity_type):
        """
        Remove the given entity type for integration of a given property.
        """
        cdef dict ip = self.information.get_dict(self.INTEGRATION_PROPERTIES)
        cdef dict prop_info = ip.get(prop_name)

        if prop_info is None:
            logger.warn('No such property %s'%(prop_name))
            return

        entity_types = prop_info.get('entity_types')
        
        if entity_types is None:
            return

        if entity_types.count(entity_type) == 1:
            entity_types.remove(entity_type)
        elif entity_types.count(entity_type) > 1:
            msg = 'Same entity included multiple times'
            logger.warn(msg)
            raise SystemError, msg            

    cpdef add_property(self, str prop_name, list integrand_arrays, list
                       integral_arrays, list entity_types=[], dict steppers={}):
        """
        Adds a new property that has to be integrated to the
        INTEGRATION_PROPERTIES dictionary in the components information.

        If the property exists already, the arrays are replaced with the new
        ones, and a warning logged.

        **PARAMETERS**
            
            - prop_name - name of the property to be be stepped.
            - integrand_arrays - names of arrays making up the integrand of this
              property. 
            - integral_arrays - names of the arrays making up the integral of
              this property.
            - entity_types - a list of accepeted entity types for this
              property. If an empty list is provided, all entities will be
              accepted. 
            - steppers - Optional information about the stepper class to use
              while stepping this property for each entity. Defaults will be
              used if this information is not present.

        Format of information to specified as part of 'steppers'
        
            - 'default' - the default stepper to use if no other information is
              available for a particular case.
            - one entry per entity type specifying the kind on stepper to use
              for this property.

        The reason for providing for such detailed information as above is to
        attain as much flexibility as needed. 
        - If each property of each entity needs a specific type of stepper,
          that can be provided using the above mechanism.
        - If each entity needs one kind of stepper irrespective of the property
          being integrated, that can be done.
        - If a single stepper can be used for any entity and any property that
          can also be done.
        - Also, if no stepper information is provided at all, defaults from the
          integrator class will be used.        

        """
        cdef dict ip = self.information.get_dict(self.INTEGRATION_PROPERTIES)
        cdef list io = self.information.get_list(self.INTEGRATION_ORDER)
        cdef dict p_dict = ip.get(prop_name)
        
        if p_dict is None:
            p_dict = dict()
            ip[prop_name] = p_dict
        else:
            logger.warn('Property %s already exists, overwriting'%(prop_name))
        
        p_dict['integrand'] = []
        p_dict['integral'] = []
        p_dict['entity_types'] = []
        p_dict['integrand'][:] = integrand_arrays
        p_dict['integral'][:] = integral_arrays
        p_dict['entity_types'][:] = entity_types        
        p_dict['steppers'] = {}
        p_dict['steppers'].update(steppers)
        
        # also append the property name to the integraion_order information.
        if io.count(prop_name) == 0:
            io.append(prop_name)
        elif io.count(prop_name) == 1:
            # we do not do anything to the order 
            # but display a warning.
            logger.warn(
                'Property %s already exists. Integration order unchanged'%(
                    prop_name)
                )
        self.setup_done = False

    cpdef add_component(self, str prop_name, str comp_name, bint pre_step=True):
        """
        Adds a component to be executed before or after the property is stepped.

        **Parameters**
        
            - prop_name - the property for whom a component is to be added.
            - comp_name - name of the component to be added. The actual
              component will be got from the component manager. 
            - pre_step - if True, add the component before the property is
              stepped, else add the component after the property is stepped.

        """
        cdef dict ip = self.information.get_dict(self.INTEGRATION_PROPERTIES)
        cdef dict p_dict = ip.get(prop_name)
        cdef list prop_components
        
        if p_dict is None:
            logger.error('Property %s does not exist'%(prop_name))
            raise ValueError, 'Property %s does not exist'%(prop_name)
        
        if pre_step == True:
            prop_components = p_dict.get('pre_step_components')
            if prop_components is None:
                prop_components = []
                p_dict['pre_step_components'] = prop_components
        else:
            prop_components = p_dict.get('post_step_components')
            if prop_components is None:
                prop_components = []
                p_dict['post_step_components'] = prop_components

        prop_components.append(comp_name)
        self.setup_done = False
        
    cpdef add_pre_integration_component(self,  str comp_name, bint
                                        at_tail=True):
        """
        Adds a component to be executed before integration of any property can
        begin.

        **Parameters**
            
            - comp_name - name of the component to be added. The actual
            component will be got from the component manager.
            - at_tail - if True, this component will be appended to the list of
            pre integration components already present. Otherwise it will be
            prepended to the said list.

        """
        cdef list pic = self.information.get_list(
            self.PRE_INTEGRATION_COMPONENTS)
        
        if at_tail ==  False:
            pic.insert(0, comp_name)
        else:
            pic.append(comp_name)

        self.setup_done = False

    cpdef set_integration_order(self, list order):
        """
        Sets the order of integration of the properties.

        **Parameters**
        
            - order - a list containg the properties in the required integration
            order.

        """
        cdef list io = self.information.get_list(self.INTEGRATION_ORDER)
        
        # first check if the new order list has the same components as the
        # current list. If not, we warn about extra / removed components, but
        # changes the order anyways.
        cdef set curr = set(io)
        cdef set new = set(order)
        
        if curr != new:
            msg = 'Current integration order and new integration order'
            msg += '\nhave different set of properties'
            logger.warn(msg)

        io[:] = order
        self.setup_done = False
    
    def setup_component(self):
        """
        Sets up the component for execution.

        The default functions sets up an explicit euler integration scheme. All
        integrand values from the previous step are used. And new values (the
        integrated values) are copied after the integration is complete.

        **ALGORITHM**
        
            - add all the pre-integration components to the execute list.
            
            - for each property in the property order list
                get property details from the INTEGRATION_PROPERTIES dict.
                for each entity in the entity list
                    if this entities type appears in the types of entities to be
                    considered for integration of this property
                        check if this entity has requested this property to be
                        integrated, (using the INTEGRATION_PROPERTIES dict of
                        the entity), if yes
                            check if there is a stepper for this property for
                            this kind of entity, if yes, create an instance of
                            that.
                            if not, create an instance of the default stepper to
                            be used for this integrator.
                            assign proper variables to the stepper and continue.
                            
                            make a note of this stepper, to later create a
                            swapper of these properties.                
                            
        """

        if self.setup_done:
            return

        prop_steppers = {}

        if len(self.execute_list) > 0:
            logger.warn('Component seems to have been setup already')
            logger.warn('That setup will be lost now')

        self.execute_list[:] = []
        
        # add the pre-integration components.
        pic = self.information.get_list(self.PRE_INTEGRATION_COMPONENTS)
        for c_name in pic:
            c = self.cm.get_component(c_name)
            if c is not None:
                self.execute_list.append(c)

        # now start appending components for the properties.
        io = self.information.get_list(self.INTEGRATION_ORDER)
        ip = self.information.get_dict(self.INTEGRATION_PROPERTIES)

        for p_name in ip:
            stepper_list = self._setup_property(p_name)
            prop_steppers[p_name] = stepper_list

        # now add copier to copy the values from the _next arrays to the main
        # arrays.     
        self._setup_copiers(prop_steppers)        
        
        self.setup_done = True

    def _setup_property(self, prop_name):
        """
        Setup the components for stepping the given property.

        **Parameters**
        
            - prop_name - name of property to setup.
            - prop_entity_dict - a dictionary to maintain, 

        **Return value**
        
            - returns the steppers created for entity for which this property
            had to be stepped. This is useful in setting up copiers for copying
            the newly computed values back to the original arrays.
        """
        prop_steppers = []

        ip = self.information.get_dict(self.INTEGRATION_PROPERTIES)
        p_info = ip.get(prop_name)
        
        if p_info is None:
            logger('No such property, %s'%(prop_name))
            return prop_steppers

        e_types = p_info['entity_types']
        
        # get the pre-step components
        psc = p_info['pre_step_components']
        for psc_name in psc:
            c = self.cm.get_component(psc_name)
            if c is not None:
                self.execute_list.append(c)
        
        # now for every entity add an appropriate stepper.
        for e in self.entity_list:
            e_ip = e.information.get_list(e.INTEGRATION_PROPERTIES)
            # check if the property spec had some entity_type information.
            accept = False
            if len(e_types) > 0:
                for e_type in e_types:
                    if e.is_a(e_type):
                        accept = True
                        break
                else:
                    accept = True

            if accept == False:
                msg = 'Entity type (%d) not accepted for %s'%(e.type, prop_name)
                logger.info(msg)
                continue        

            # check if the entity itself has some integration information.
            if e_ip is not None:
                if (len(e_ip) == 0 or (not prop_name in e_ip)):
                    # entity either does not want this property to be
                    # integrated or does not want anything to be integrated
                    # continue without doing anything for this entity.
                    continue
                
            # add a stepper for this property for this entity.
            # get an appropriate stepper class for this property for this
            # entity type.
            stepper = self.get_stepper(e.type, prop_name)
            if stepper is not None:
                stepper.add_entity(e)
                prop_steppers.append(stepper)
                stepper.setup_component()
                self.execute_list.append(stepper)
            else:
                msg = 'No stepper found for entity type (%d) for property%s'%(e.type, prop_name)
                logger.error(msg)

        return prop_steppers
        
    def get_stepper(self, entity_type, prop_name):
        """
        Returns a stepper to be used for this kind of entity and of this
        property.

        **Parameters**
        
            - entity_type - an integer specifying the entities type.
            - prop_name - a string specifying the property_name.

        **Algorithm**::

            - check if the property specified is correct, return None otherwise
            
            - check if the property has any particular stepper requirements.
                - if no steppers information is provided for the property OR
                - if an empty dictionary was provided for steppers OR
                - if a dictionary without the entity type and without a default
                was provided
                THEN
                Use stepper defaults from the integrator.
            - ELSE get the stepper for the property spec.

        """
        ip = self.information.get_dict(self.INTEGRATION_PROPERTIES)
        default_steppers = self.information.get_dict(self.DEFAULT_STEPPERS)

        # check if the property is valid.
        if not ip.has_key(prop_name):
            return None
        
        # check if any stepper specification is found for the property.
        steppers = ip.get('steppers')

        if (steppers is None or len(steppers) == 0 or
            (not steppers.has_key(entity_type) and not
             steppers.has_key('default'))):

            if default_steppers.has_key(entity_type):
                return cfac.get_component('ode_stepper',
                                          default_steppers[entity_type])
            elif default_steppers.has_key('default'):
                return cfac.get_component('ode_stepper',
                                          default_steppers['default'])
            else:
                logger.warn('No default stepper provided for integrator')
                return None
        else:
            # meaning, the property has some stepper specification.
            if steppers.has_key(entity_type):
                return cfac.get_component('ode_stepper',
                                          steppers[entity_type])
            else:
                return cfac.get_component('ode_stepper',
                                          steppers['default'])

        return None

    def _setup_copiers(self, prop_stepper_dict):
        """
        Create copiers for each property, to copy the next step values to the
        current arrays.
        """
        ip = self.infomration.get_dict(self.INTEGRATION_PROPERTIES)

        for prop_name in prop_stepper_dict.keys():
            stepper_list = prop_stepper_dict[prop_name]
            prop_info = ip.get(prop_name)
            # create a copier for each stepper in the stepper_list.
            for stepper in stepper_list:
                cop_name = 'copier_'+prop_name
                copier = cfac.get_component('copier', 'copier', 
                                            cop_name, stepper.entity_list,
                                            stepper.next_step_names,
                                            stepper.integral_arrays)
                self.execute_list.append(copier)
        
    cdef int compute(self) except -1:
        """
        Perform the integraiton.
        """
        cdef int i, n_comps
        cdef SolverComponent comp
        # make sure the component is setup properly
        self.setup_component()

        n_comps = len(self.execute_list)
        
        for i from 0 <= i < n_comps:
            comp = self.execute_list[i]
            comp.compute()
        
    cpdef int update_property_requirements(self) except -1:
        """
        Updates the property requirements of the integrator.

        All properties required will be considered as WRITEable PARTICLE
        properties.
        """
        pass
