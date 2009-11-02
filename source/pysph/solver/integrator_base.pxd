"""
Base class for all integrators.

**Member variables**

    - integrated_properties - the list of properties that need to be integrated.
    - execute_list - list of components that will be executed on every call to
      integrate.

**Notes**
    
    - Some of the features needed of this class:

        - Specify any number of properties to be integrated.
        - Specify / modify the order in which they are integrated.
        - Insert components to be executed, between various the integration
          components. 
        - Extending from comments in the integration_step.pxd file, say we want
          to implement the following integrator:

          - properties to be integrated (in the order given)
            - acceleration - ax, ay, az
            - velocity - vx, vy, vz
            - density rate - rho_rate.

---------

          - following components are to be exectued before any operation:
            (in the order given)
            - boundary updates.
            - dummy particle updates (for the viscosity component).
            - density compute
          
          - acceleration is computed by using the following components:
            - pressure gradient.
            - viscosity.
            - boundary forces.
            - acceleration of moving solids is computed.
            
          - after acceleration integration,  the following
            components are to be exectued to update the velocity obtained as a
            result of integrating acceleration:
              - an XSPH velocity correction to get smooth velocities.
              - A pressure correction method to obtain zero divergence.

          - A collision handling mechanism is to be executed using the
            velocities computed by the acceleration integrator.

          - The velocities computed by the collision handler are then
            integrated to get new positions.

--------

The above steps represent one step of the integrator. For a runge kutta
integrator, again, the same operations are to be done, execpt some of the
integration steps may use different variables. I need to encapsulate it
properly.

"""

cdef class ODEStep(SolverComponent):
    """
    Move forward a variable (exactly one ?) by a time-step.
    """
    pass

cdef class Integrator(SolverComponent):
    """
    """
    # the final list of components that will be executed at every call to
    # compute().
    cdef public execute_list

    # list of properties that need to be integrated.
    cdef public dict integration_properties

    # list of components that are to be executed before integration of any
    # property can be done.
    cdef public dict pre_integration_components

    # contains one list per property, indicating the list of components that are
    # to be executed before integrating this property.
    cdef public dict per_property_components

    # add an entity whose properties have to be integrated.
    cpdef add_entity(self, EntityBase entity)

    # add a component to be executed before integration of this property.
    cpdef add_component(self, str property, str comp_name, bint at_tail=True)
    
    # add a component to be exectued before integration of any property is done.
    cpdef add_pre_integration_component(self, str comp_name, bint
                                        at_tail=True)
    
    # set the order in which properties should be integrated.
    cpdef set_integration_order(self, dict order)
    
    cdef int compute(self) except -1
    
    # setup the component once prior to execution.
    cpdef setup_component(self) except -1
    
cdef class AccelerationVelocityIntegrator(Integrator):
    """
    Base class for all integrators implementing basic acceleration-velocity
    integration.
    """
    cdef 
    pass

cdef class ImplicitEulerIntegrator(Integrator):
    """
    """
    def __cinit__(self, *properties):
        """
        """
        self.integration_properties['velocity'] = None
        self.integration_properties['acceleration'] = None

        for prop in properties:
            self.integration_properties[prop] = None

        


    cdef int compute(self) except -1

cdef class ExplicitEulerIntegrator(Integrator):
    """
    """
    pass

cdef class RungeKutta2Integrator(Integrator):
    """
    """
    cdef int compute(self) except -1

cdef class RungeKutta4Integrator(Integrator):
    """
    """
    pass

cdef class VerletIntegrator(Integrator):
    """
    """
    pass

    
