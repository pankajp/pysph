"""
Contains base classes for all integrators.
"""

# logger import
import logging
logger = logging.getLogger()

# numpy imports
cimport numpy

# local imports
from pysph.base.attrdict import AttrDict
from pysph.base.carray cimport DoubleArray
from pysph.base.particle_array cimport ParticleArray
from pysph.solver.solver_base cimport SolverComponent, SolverBase,\
    ComponentManager
from pysph.solver.entity_base cimport EntityBase
from pysph.solver.solid import Solid
from pysph.solver.fluid import Fluid
from pysph.solver.time_step cimport TimeStep
from pysph.solver.array_initializer import ArrayInitializer
from utils import *

# import component factory
from pysph.solver.component_factory import ComponentFactory as cfac


###############################################################################
# `ODEStepper` class.
###############################################################################
cdef class ODEStepper(SolverComponent):
    """
    Class to step a given property by a given time step.

    This class implements a simple euler step. The values of the next time step
    are stored in arrays, whose names are got by suffixing _next to the original
    array name.

    """
    category='ode_stepper'
    identifier = 'base'
    def __cinit__(self, str name='', SolverBase solver=None, ComponentManager
                  component_manager=None, list entity_list=[], str prop_name='',
                  list integrands=[], list integrals=[], TimeStep
                  time_step=None, *args, **kwargs):
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

        if self.setup_done == True:
            return 0

        # make sure timestep is setup properly
        if self.time_step is None:
            raise ValueError, 'time_step not set'

        self.next_step_names[:] = []
        
        num_props = len(self.integral_names)
        
        for i in range(num_props):
            arr_name = self.integral_names[i]
            self.next_step_names.append(
                arr_name + '_next')

        # now make sure that all the entities have the _next property.
        num_entities = len(self.entity_list)
        for i in range(num_entities):
            e = self.entity_list[i]
            logger.debug('Setting up %s'%(e.name))
            parr = e.get_particle_array()
            if parr is not None:
                # make sure all the _next arrays are present, if not add
                # them.
                for j from 0 <= j < num_props:
                    arr_name = self.next_step_names[j]
                    logger.debug('adding property %s '%(arr_name))
                    if not parr.properties.has_key(arr_name):
                        parr.add_property({'name':arr_name})
            else:
                logger.warn('No parr of entity (%s)'%(e.name))
                logger.warn('Removing from entity_list')
                to_remove.append(e)

        # remove entities that did not provide a particle array.
        for e in to_remove:
            self.entity_list.remove(e)
            
        # mark component setup as done.
        self.setup_done = True
            
        return 0
        
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

        for i in range(num_entities):
            e = self.entity_list[i]
            
            parr = e.get_particle_array()
    
            if parr is None:
                logger.warn('no particle array for %s'%(e.name))
                continue
            
            for j from 0 <= j < num_props:
                an = parr._get_real_particle_prop(self.integral_names[j])
                bn = parr._get_real_particle_prop(self.integrand_names[j])
                an1 = parr._get_real_particle_prop(self.next_step_names[j])
                
                an1[:] = an + bn*self.time_step.value

        return 0
###############################################################################
# `PyODEStepper` class.
###############################################################################
cdef class PyODEStepper(ODEStepper):
    """
    Simple class to implement some ODEStepper in python.
    """
    category='ode_stepper'
    identifier='py_base'
    def __cinit__(self, name='', SolverBase solver=None, ComponentManager
                  component_manager=None, entity_list=[], 
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
###############################################################################
# `StepperInfo` class.
###############################################################################
cdef class StepperInfo:
    """
    Contains information about stepping a particular property.
    """
    def __cinit__(self, str property_name, 
                  list y_arrays=[], list dydt_arrays=[], 
                  list pre_step_components=[], list post_step_components=[],
                  steppers={}):
        """
        """
        pass
    
###############################################################################
# `Integrator` class.
###############################################################################
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
    PRE_STEP_COMPONENTS = 'PRE_STEP_COMPONENTS'
    POST_STEP_COMPONENTS = 'POST_STEP_COMPONENTS'
    PRE_INTEGRATION_COMPONENTS = 'PRE_INTEGRATION_COMPONENTS'
    POST_INTEGRATION_COMPONENTS = 'POST_INTEGRATION_COMPONENTS'
    DEFAULT_STEPPERS = 'DEFAULT_STEPPERS'
    INTEGRATION_ORDER = 'INTEGRATION_ORDER'
    
    identifier = 'euler'
    category = 'integrator'

    def __cinit__(self, str name='', SolverBase solver=None, 
                  ComponentManager component_manager=None, 
                  list entity_list=[], 
                  int dimension=3,
                  *args, **kwargs):
        """
        Constructor.
        """
        self.execute_list = []

        if solver is None:
            self.time_step = TimeStep()
        else:
            self.time_step = solver.time_step
            
        self.information = AttrDict()
        self.information[self.INTEGRATION_PROPERTIES] = {}
        self.information[self.PRE_INTEGRATION_COMPONENTS] = []
        self.information[self.INTEGRATION_ORDER] = []
        self.information[self.DEFAULT_STEPPERS] = {}
        self.information[self.POST_INTEGRATION_COMPONENTS] = []
        self.information[self.PRE_STEP_COMPONENTS] = []
        self.information[self.POST_STEP_COMPONENTS] = []

        #self.setup_defualt_steppers()
        # setup the velocity and position properties according to dimension
        self.set_dimension(dimension)

    def __init__(self, name='', solver=None, component_manager=None,
                 entity_list=[], dimension=3, *args, **kwargs):
        """
        """
        #pass
        self.setup_defualt_steppers()

    def setup_defualt_steppers(self):
        """
        """
        cdef dict default_steppers = self.information[self.DEFAULT_STEPPERS] 
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
            logger.warn('Not adding velocity and position properties')
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
                          [EntityBase], integrand_initial_values=[0.])
        self.add_property('position', vel_arrays, pos_arrays,
                          [EntityBase])

    cpdef add_entity_type(self, str prop_name, type entity_type):
        """
        Includes the given entity type for integration of a given property.
        """
        cdef dict ip = self.information[self.INTEGRATION_PROPERTIES]
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

    cpdef remove_entity_type(self, str prop_name, type entity_type):
        """
        Remove the given entity type for integration of a given property.
        """
        cdef dict ip = self.information[self.INTEGRATION_PROPERTIES]
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

    cpdef dict get_property_step_info(self, str prop_name):
        """
        Returns the dict associated with the property prop_name, containing the
        stepping information of the said property.
        """
        cdef dict ip = self.information[self.INTEGRATION_PROPERTIES]
        
        return ip.get(prop_name)

    def add_property_step_info(self, prop_name, list integrand_arrays, list
                               integral_arrays, list entity_types=[], dict
                               steppers={}, integrand_initial_values=None):
        """
        """
        # FIXME
        # remove the add_property function and make this the primary function.
        self.add_property(prop_name=prop_name,
                          integrand_arrays=integrand_arrays,
                          integral_arrays=integral_arrays,
                          entity_types=entity_types,
                          steppers=steppers,
                          integrand_initial_values=integrand_initial_values)        

    cpdef add_property(self, str prop_name, list integrand_arrays, list
                       integral_arrays, list entity_types=[], dict steppers={},
                       list integrand_initial_values=None):
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
            - integrand_initial_values - An array having one value for each
              array in the integrand array. If this is present, an array
              initializer will be added as the first component of the pre-step
              component of this property.

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
        cdef dict ip = self.information[self.INTEGRATION_PROPERTIES]
        cdef list io = self.information[self.INTEGRATION_ORDER]
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
        if integrand_initial_values is not None:
            p_dict['integrand_initial_values'] = []
            p_dict['integrand_initial_values'][:] = integrand_initial_values
        else:
            p_dict['integrand_initial_values'] = None
        
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

    cpdef add_pre_step_component(self, str comp_name, str property_name='',
                                 bint at_tail=True):
        """
        Adds a component to be executed before stepping of the property given by
        property_name. If property_name name is '', the component is added at a
        point before stepping of any property is done.
        """
        cdef dict ip = self.information[self.INTEGRATION_PROPERTIES]
        cdef list psc = self.information[self.PRE_STEP_COMPONENTS]
        cdef dict p_dict = ip.get(property_name)
        cdef list prop_components

        if property_name == '':
            if psc.count(comp_name) == 0:
                if at_tail == True:
                    psc.append(comp_name)
                else:
                    psc.insert(0, comp_name)
        else:
            if p_dict is None:
                logger.error('Property %s does not exist'%(property_name))
                raise ValueError, 'Property %s does not exist'%(property_name)
        
            prop_components = p_dict.get('pre_step_components')

            if prop_components is None:
                prop_components = []
                p_dict['pre_step_components'] = prop_components
            
            if at_tail:
                prop_components.append(comp_name)
            else:
                prop_components.insert(0, comp_name)

        self.setup_done = False

    cpdef add_post_step_component(self, str comp_name, str property_name='',
                                  bint at_tail=True):
        """
        Adds a component to be executed after stepping of the property given by
        property_name. If the property_name is '', the component is added after
        stepping of all properties are done.
        """
        cdef dict ip = self.information[self.INTEGRATION_PROPERTIES]

        cdef list psc = self.information[self.POST_STEP_COMPONENTS]

        cdef dict p_dict = ip.get(property_name)
        cdef list prop_components

        if property_name == '':
            if psc.count(comp_name) == 0:
                if at_tail == True:
                    psc.append(comp_name)
                else:
                    psc.insert(0, comp_name)
        else:
            if p_dict is None:
                logger.error('Property %s does not exist'%(property_name))
                raise ValueError, 'Property %s does not exist'%(property_name)
        
            prop_components = p_dict.get('post_step_components')

            if prop_components is None:
                prop_components = []
                p_dict['post_step_components'] = prop_components
            
            if at_tail == True:
                prop_components.append(comp_name)
            else:
                prop_components.insert(0, comp_name)
                
        self.setup_done = False 

    cpdef add_pre_integration_component(self,  str comp_name, bint
                                        at_tail=True):
        """
        Adds a component to be executed before integration of any property can
        begin. For multi-step integrators, this should add a component before
        the first step begins. It should _NOT_ be inserted before every step.

        **Parameters**
            
            - comp_name - name of the component to be added. The actual
            component will be got from the component manager.
            - at_tail - if True, this component will be appended to the list of
            pre integration components already present. Otherwise it will be
            prepended to the said list.

        """
        cdef list pic = self.information[self.PRE_INTEGRATION_COMPONENTS]
        
        if at_tail ==  False:
            pic.insert(0, comp_name)
        else:
            pic.append(comp_name)

        self.setup_done = False

    cpdef add_post_integration_component(self, str comp_name, bint
                                         at_tail=True):
        """
        Adds a component to be executed after integration of all properties has
        been done. For multi-step integrators, this should add a component after
        all steps of the integrator are done. This should _NOT_ be called after
        each step is done.

        **Parameters**
        
            - comp_name - name of the component to be added. The actual
            component will be got from the component manager.
            - at_tail - if True the component will be appended to the list of
            post integration components already present. Otherwise if will be
            prepended to the list of pre integration components.

        """
        cdef list pic = self.information[self.POST_INTEGRATION_COMPONENTS]

        if at_tail == False:
            pic.insert(0, comp_name)
        else:
            pic.append(comp_name)

        self.setup_done = False
    
    cpdef set_integration_order(self, list order):
        """
        Sets the order of integration of the properties.

        **Parameters**
        
            - order - list containing the properties in the required integration
            order.

        """
        cdef list io = self.information[self.INTEGRATION_ORDER]
        
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
        
    cpdef int setup_component(self) except -1:
        """
        Sets up the component for execution.

        The default functions sets up an explicit euler integration scheme. All
        integrand values from the previous step are used. And new values (the
        integrated values) are copied after the integration is complete.

        """
        if self.setup_done:
            return 0

        if len(self.execute_list) > 0:
            logger.warn('Component seems to have been setup already')
            logger.warn('That setup will be lost now')

        # clear the execute list.
        self.execute_list[:] = []
        
        # add the pre-integration components.
        self._setup_pre_integration_components()

        # setup the integration step.
        self._setup_step()

        # now add any post integration components to be executed. For one step
        # integrators post-step and post integration components are the same.
        self._setup_post_integration_components()

        self.setup_done = True
        
        return 0

    def _setup_pre_integration_components(self):
        """
        Add the pre-integration components to the execute list.
        """
        # add the pre-integration components.
        pic = self.information[self.PRE_INTEGRATION_COMPONENTS]
        for c_name in pic:
            c = self.cm.get_component(c_name)
            if c is not None:
                self.execute_list.append(c)

    def _setup_post_integration_components(self):
        """
        Add the post-integration components to the execute list.
        """
        # add the post-integration components.
        pic = self.information[self.POST_INTEGRATION_COMPONENTS]
        for c_name in pic:
            c = self.cm.get_component(c_name)
            if c is not None:
                self.execute_list.append(c)
            else:
                logger.error('Post integration component %s not found'%(
                        c_name))

    def _setup_pre_stepping_components(self):
        """
        Setup components that are to be executed before stepping of any property
        beings. 
        """
        psc = self.information[self.PRE_STEP_COMPONENTS]
        for c_name in psc:
            c = self.cm.get_component(c_name)
            if c is not None:
                self.execute_list.append(c)

    def _setup_post_stepping_components(self):
        """
        Setup components that are to be executed after stepping of all
        properties is done and new values are copied.
        """
        psc = self.information[self.POST_STEP_COMPONENTS]
        for c_name in psc:
            c = self.cm.get_component(c_name)
            if c is not None:
                self.execute_list.append(c)

    def _setup_property_copiers(self, prop_stepper_dict):
        """
        """
        self._setup_copiers(prop_stepper_dict)

    def _setup_property_steppers(self):
        """
        """
        prop_steppers = {}

        # now start appending components for the properties.
        io = self.information[self.INTEGRATION_ORDER]

        for p_name in io:
            stepper_list = self._setup_property(p_name)
            prop_steppers[p_name] = stepper_list

        return prop_steppers
        
    def _setup_step(self):
        """
        """
        self._setup_pre_stepping_components()

        prop_steppers = self._setup_property_steppers()

        self._setup_property_copiers(prop_steppers)

        self._setup_post_stepping_components()

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

        ip = self.information[self.INTEGRATION_PROPERTIES]
        p_info = ip.get(prop_name)
        
        if p_info is None:
            logger('No such property, %s'%(prop_name))
            return prop_steppers

        e_types = p_info.get('entity_types')

        # setup initializers for each entity that requires this property to be
        # stepped. 
        integrand_initial_values = p_info.get('integrand_initial_values')
        if integrand_initial_values != None:
            e_list = []
            for e in self.entity_list:
                e_ip = e.information[e.INTEGRATION_PROPERTIES]
                if len(e_types) > 0:
                    if e.is_type_included(e_types):
                        if e_ip is not None:
                            if len(e_ip) == 0 or not(prop_name in e_ip):
                                continue
                        else:
                            e_list.append(e)
            # we have the list of entities that are to be considered for 
            # the integration of this property.
            ai_name=extract_entity_names(e_list)
            logger.info('Adding initializer for %s'%(prop_name))
            logger.info('Initial values : %s'%(integrand_initial_values))
            logger.info('Integrand is : %s'%(p_info.get('integrand')))
            ai = ArrayInitializer(name='init_'+ai_name,
                                  solver=self.solver,
                                  entity_list=e_list,
                                  array_names=p_info.get('integrand'),
                                  array_values=integrand_initial_values)
            self.execute_list.append(ai)
        
        # get the pre-step components
        psc = p_info.get('pre_step_components')
        if psc is not None:
            for psc_name in psc:
                c = self.cm.get_component(psc_name)
                if c is not None:
                    self.execute_list.append(c)
        
        # now for every entity add an appropriate stepper.
        for e in self.entity_list:
            e_ip = e.information[e.INTEGRATION_PROPERTIES]
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
                msg = 'Entity type (%s) not accepted for %s'%(str(e.type), prop_name)
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
                stepper.name = 'stepper_'+prop_name+'_'+e.name
                stepper.add_entity(e)
                prop_steppers.append(stepper)
                stepper.time_step = self.time_step
                stepper.setup_component()
                self.execute_list.append(stepper)
            else:
                msg = 'No stepper found for entity type'
                msg += ' (%s) for property%s'%(str(e.type), prop_name)
                logger.error(msg)

        # setup the post step components
        psc = p_info.get('post_step_components')
        if psc is not None:
            for psc_name in psc:
                c = self.cm.get_component(psc_name)
                if c is not None:
                    self.execute_list.append(c)

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
        ip = self.information[self.INTEGRATION_PROPERTIES]
        default_steppers = self.information[self.DEFAULT_STEPPERS]
        
        prop_info = ip.get(prop_name)
        # check if the property is valid.
        if prop_info is None:
            logger.warn('Property name incorrect')
            return None

        integrand = prop_info.get('integrand')
        integral = prop_info.get('integral')

        logger.debug('Adding stepper for %s, %s'%(
                str(integrand), str(integral)))

        if integral is None or integrand is None:
            logger.warn('integrand or integral is None')
            return None

        # check if any stepper specification is found for the property.
        steppers = prop_info.get('steppers')
        stepper_type = ''

        if (steppers is None or len(steppers) == 0 or
            (not steppers.has_key(entity_type) and not
             steppers.has_key('default'))):
            if default_steppers.has_key(entity_type):
                stepper_type = default_steppers[entity_type]
            elif default_steppers.has_key('default'):
                stepper_type = default_steppers['default']
        else:
            # meaning, the property has some stepper specification.
            if steppers.has_key(entity_type):
                stepper_type = steppers[entity_type]
            else:
                stepper_type = steppers['default']
        
        if stepper_type == '':
            logger.warn('No default stepper provided for integrator')
            return None
        else:
            return cfac.get_component('ode_stepper',
                                      stepper_type,
                                      solver=self.solver,
                                      prop_name=prop_name,
                                      integrands=integrand,
                                      integrals=integral,
                                      time_step=self.time_step)
                    
    def _setup_copiers(self, prop_stepper_dict):
        """
        Create copiers for each property, to copy the next step values to the
        current arrays.
        """
        ip = self.information[self.INTEGRATION_PROPERTIES]
        io = self.information[self.INTEGRATION_ORDER]

        for prop_name in io:
            stepper_list = prop_stepper_dict[prop_name]
            # create a copier for each stepper in the stepper_list.
            for stepper in stepper_list:
                e_names = extract_entity_names(stepper.entity_list)
                cop_name = 'copier_'+prop_name+'_'+e_names
                copier = cfac.get_component('copiers', 'copier', 
                                            name=cop_name, 
                                            solver=self.solver,
                                            entity_list=stepper.entity_list,
                                            from_arrays=stepper.next_step_names,
                                            to_arrays=stepper.integral_names)
                if copier is None:
                    msg = 'Could not create copier for %s'%(prop_name)
                    logger.warn('Could not create copier for %s'%(prop_name))
                    raise SystemError, msg

                logger.info('Adding copier : %s'%(copier))
                logger.info('From : %s to %s'%(copier.from_arrays, copier.to_arrays))
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
        
        for i in range(n_comps):
            comp = self.execute_list[i]
            comp.compute()

        return 0

    cpdef int integrate(self) except -1:
        """
        cdefed wrapper for the compute function.
        """
        return self.compute()

    cpdef int update_property_requirements(self) except -1:
        """
        Updates the property requirements of the integrator.

        All properties required will be considered as WRITEable PARTICLE
        properties.
        """
        cdef str prop_name
        cdef dict ip = self.information[self.INTEGRATION_PROPERTIES]
                
        for prop_name in ip.keys():
            prop_info = ip.get(prop_name)
            intgnd = prop_info.get('integrand')
            intgl = prop_info.get('integral')
            
            e_types = prop_info.get('entity_types')

            if e_types is None or len(e_types) == 0:
                # meaning this property is to be applied to all entity types. 
                # add the integrand and integral arrays
                for i in range(len(intgnd)):
                    self.add_write_prop_requirement(EntityBase, 
                                                    intgnd[i])
                    self.add_write_prop_requirement(EntityBase,
                                                    intgl[i])
            else:
                for e_type in e_types:
                    for i in range(len(intgnd)):
                        self.add_write_prop_requirement(e_type, intgnd[i])
                        self.add_write_prop_requirement(e_type, intgl[i])
                        
        return 0
