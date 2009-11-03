"""
Contains base classes for all integrators.
"""

# local imports
from pysph.solver.solver_component cimport SolverComponent, ComponentManager
from pysph.solver.entity_base cimport EntityBase


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
        For every property to be integrated, a integer specifying its position
        in the list of properties. Implemented as a dict keyed on the property
        names.

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
    
    def __cinit__(self, str name='', ComponentManager cm=None, *args, **kwargs):
        """
        Constructor.
        """
        self.information.set_dict(self.INTEGRATION_PROPERTIES, {})
        self.information.set_dict(self.PRE_INTEGRATION_COMPONENTS, {})
        
        self.information.set_dict(self.INTEGRATION_ORDER, {})
        self.information.set_str(self.DEFAULT_STEPPER, '')
        self.information.set_dict(self.STEPPERS, {})

        
        self.entity_list = set()
        self.execute_list = []

    cpdef add_entity(self, EntityBase entity):
        """
        Adds an entity whose properties are to be integrated.
        """
        self.entity_list.add(entity)

    cpdef add_component(self, str property, str comp_name, bint pre_step=True):
        """
        Adds a component to be executed before or after the property is stepped.

        **Parameters**
        
            - property - the property for whom a component is to be added.
            - comp_name - name of the component to be added. The actual
              component will be got from the component manager. 
            - pre_step - if True, add the component before the property is
              stepped, else add the component after the property is stepped.

        """
        pass        

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
        pass

    cpdef set_integration_order(self, list order):
        """
        Sets the order of integration of the properties.

        **Parameters**
        
            - order - a list containg the properties in the required integration
            order.

        """
        pass
    
    cpdef int setup_component(self) except -1:
        """
        Sets up the component for execution.
        """
        pass
    
    cpdef int compute(self) except -1:
        """
        Perform the integraiton.
        """
        pass
