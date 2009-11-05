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
# `ODESteper` class.
################################################################################
cdef class ODESteper(SolverComponent):
    """
    Class to step a given property by a given time step.

    This class implements a simple euler step. The values of the next time step
    are stored in arrays, whose names are got by suffixing _next to the original
    array name.

    """
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

    **INTEGRATION_ORDER**
        A list of property names in the order in which they have to be stepped.

    **STEPPERS**
        For each entity type, for each property type, the kind if stepper to be
        used. This allows for a lot of flexibility in using different kinds of
        steppers for different properties for different types of entities. If a
        stepper for a property or entity is not given, the DEFAULT_STEPPER will
        be used.

    **DEFAULT_STEPPER**
        A default stepper to be used when no stepper has been specified for any
        entity type.

    """
    # list of information keys provided by this class.
    INTEGRATION_PROPERTIES = 'INTEGRATION_PROPERTIES'
    PRE_INTEGRATION_COMPONENTS = 'PRE_INTEGRATION_COMPONENTS'
    DEFAULT_STEPPER = 'DEFAULT_STEPPER'
    STEPPERS = 'STEPPERS'
    INTEGRATION_ORDER = 'INTEGRATION_ORDER'
    
    def __cinit__(self, str name='', ComponentManager cm=None, int dimension=3, *args, **kwargs):
        """
        Constructor.
        """
        self.entity_list = set()
        self.execute_list = []

        self.information.set_dict(self.INTEGRATION_PROPERTIES, {})
        self.information.set_list(self.PRE_INTEGRATION_COMPONENTS, [])
        self.information.set_list(self.INTEGRATION_ORDER, [])
        self.information.set_str(self.DEFAULT_STEPPER, '')
        self.information.set_dict(self.STEPPERS, {})

        self.set_dimension(dimension)        

    cpdef set_dimension(self, int dimension):
        """
        Sets the dimension of the velocity and position vectors.
        """
        # sanity checks
        if dimension == 0:
            dimension = 1
            logger.warn('Dimension of 0 specified, using 1')
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
            
        # now create the properties.
        self.add_property('velocity', accel_arrays, vel_arrays)
        self.add_property('position', vel_arrays, pos_arrays)

    cpdef add_entity(self, EntityBase entity):
        """
        Adds an entity whose properties are to be integrated.
        """
        if self.filter_entity(entity):
            self.entity_list.add(entity)

    cpdef add_property(self, str prop_name, list integrand_arrays, list
                       integral_arrays, list entity_types=[]):
        """
        Adds a new property that has to be integrated to the
        INTEGRATION_PROPERTIES dictionary in the components information.

        If the property exists already, the arrays are replaced with the new
        ones, and a warning logged.

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
        p_dict['integrals'] = []
        p_dict['entity_types'] = []
        p_dict['integrand'][:] = integrand_arrays
        p_dict['integrals'][:] = integral_arrays
        p_dict['entity_types'][:] = entity_types        
        
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
        cdef list pic = self.information.get_dict(
            self.PRE_INTEGRATION_COMPONENTS)
        
        if at_tail ==  False:
            pic.insert(0, comp_name)
        else:
            pic.append(comp_name)        

    cpdef set_integration_order(self, list order):
        """
        Sets the order of integration of the properties.

        **Parameters**
        
            - order - a list containg the properties in the required integration
            order.

        """
        cdef list io = self.information.get_dict(self.INTEGRATION_ORDER)
        
        # first check if the new order list has the same components as the
        # current list. If not, we warn about extra / removed components, but
        # changes the order anyways.
        cdef set curr = set(io)
        cdef set new = set(order)
        
        if curr != new:
            msg = 'Current integration order and new integration order'
            msg += ' have different set of properties'
            logger.warn(msg)

        io[:] = order            
    
    cpdef int setup_component(self) except -1:
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
        pass
    
    cpdef int compute(self) except -1:
        """
        Perform the integraiton.
        """
        pass
