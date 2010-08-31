"""
Class to represent one integration step.

This class abstracts various operations that need to be performed in any
integration step. For single step integrators, one concrete class of this kind
should suffice. Multi-step integrators may need to implement more than one class
of this kind, depending on their requirements. A list of operations that should
be performed is given below: 

    - Apply boundary conditions. This includes the following:
        
        - removing particles that have reached outflow boundaries.
        - adding particles from inflows.
        - ...

    - Update/add/remove dummy particles.

        - some SPH algorithms need to add dummy particles. For example, the
          ghost particles that are added for each fluid particle near a solid wall
          to achieve no penetration through the wall. 
        - Another example is the implementation of periodic boundary
          conditions. Here particles will need to be mirrored from the `other'
          side of the periodic boundary, before any computations can be done.
        - A paper by ferrai et. al. needs to add dummy particles for every
          boundary particle that is near a given fluid particle.
        - Thus the requirement of adding dummy particles is fairly
          ubiquitous. Many algorithms will need it. Hence this step seems valid.
        - A special case. What if some algorithm wants to add some dummy
          particles in every loop of the SPH calculation ? Meaning that in the
          SPH summation, once it has found all the required neighbors, it adds
          some particles and does some specific operations. This can again be
          implemented in two ways. You could implement a special SPH class
          derived from the SPHBase, or you could make that algorithm add the
          dummy particles at this stage of integration and label them
          appropriately so that only they are used in the SPH summation as
          neighbors. Again you will need to implement a separate SPH summation
          class, but there you will just check if a particle has a particular
          label before including it in the computation.

    - Compute particle properties - density, interaction radius, pressure etc.

    - Compute forces on all particles.

        - Forces on all sorts of particles have to be computed at this
          step. This could involve the following:
              
              - forces due to interaction among fluid particles - pressure,
                viscosity.
              - forces due to interaction of fluid particles with solid
                particles - viscosity, boundary repulsion etc.
              - forces due to interaction among solid particles - solid
                mechanics simulations will need this.
              - forces on all kinds of particles due to programmed motions of
                objects. For example, a fan blade revolving with constant
                angular velocity. At every timestep the angular linear velocity
                of particles making up this blade will change - hence a linear
                acceleration, which will have to be computed and integrated.

    - Compute new velocities. With the accelerations with us, we can compute the
      new velocities using the given step size (may have to be maintained for
      each particle).

    - Handle collisions if needed. If a collision handling component is
      provided, collisions are handled using that component. What exactly does
      'handling collisions' means is yet to be decided. But after this step, the
      new positions computed for particles should not involve collisions. That
      is what this step should guarantee.

    - Compute new positions.


Building on the above ideas, and some extra requirements:

    - The integrator class to consist of the following:

        - list of properties that have to be integrated, for example
            - acceleration to be integrated to get velocity.
            - velocity to be integrated to get new positions.
            - rate of change of density to be integrated to get density for the
              next step.
            - rate of change of internal energy to get the next internal energy.
            - rate of change of interaction radius to get the new radii for
              particles.
            - things i am not aware of currently.
        - for each property in the list of properties to be integrated, a list
          of compoments component that is used to compute that property or
          update that property prior to integration.
        - No prior assumptions about the properties and what order in which they
          have to be integrated. This is the most generic level, and anything
          specific is to be done in derived classes.

        - There will be cases as indicated above, where some component that
          computes a particular property, also needs to do some pre-integration
          setup. The doubt was whether to include some automation to this. Does
          not seem to be a very good idea.
"""

from pysph.solver.solver_base cimport SolverComponent

cdef class ODEStepper(SolverComponent):
    """
    Class to advance a set of ODE's by one step.
    
    The ODE takes the following form:
    .. math::
        
        \frac{dA}{dt} = B(t)

    To step this ODE one step ahead in time, we integrate B(t) wrt 't' to get
    A(t+1).
    .. math::
    
        A(n+1) = A(n) + \delta t * B(n)

    All properties in the ODEStepper will be "stepped" in a call to step. So
    only group related properties in a ODEStepper. For example three components
    of velocity could be coupled here.

    **Member Variables**
    
        - integrands - the list of property names that are to be integrated.
        - integrals - the list of property names where the values after
          integration will be stored.
        - integrals_next - the list of property name where the values of the
          next step are stored.
        - particle_arrays - this list of particle arrays whose properties are
          to be integrated.

    The ODE step may be implemented differently in derived classes. For example,
    the verlet position step uses both velocity and acceleration.

    A(n+1) will be always written to a array called A_next.    

    """
    # list of properties that are to be integrated.
    # for example ['ax', 'ay', 'az'] for acceleration.
    cdef public list integrands

    # list of properties that are the integrands, the previous
    # step values will be read from these arrays.
    cdef public list integrals

    # array names to store the values of the next step.
    cdef public list integrals_next

    # the time step to use.
    cdef double time_step

    cdef int compute(self) except -1

cdef class ODEInfo:
    """
    Encapsulates information for integrating an ODE.

    The information includes:
        
        - properties involved in this ODE, i.e. the integral and the integrand. 
        - the components to be executed to compute the value of the integrand.
        - components that may need to be executed after the integration step.
        - the order in which the components are to be executed.

    """
    cdef public ODEStepper ode_stepper
    cdef public dict pre_step_components
    cdef public dict post_step_components
    cdef dict _pre_step_component_order
    cdef dict _post_step_component_order

    cpdef add_pre_step_component(self, str name, SolverComponent c, int pos=*)
    cpdef add_post_step_component(self, str name, SolverComponent c, int pos=*)

    # to copy components from another ODEInfo.
    cpdef copy_components(self, ODEInfo)
    
cdef class IntegrationStep:
    """
    Class to encapsulate one step of integration.

    This should be the 'step' used for any integrator, be it euler, rk2, rk4, verlet
    etc. 
    """    
    cdef public list execute_list

    cdef public dict pre_integration_components
    cdef public dict pic_order

    cdef public dict ode_info_dict
    cdef public dict ode_order

    cdef void step(self)
    cdef void add_ode(self, ODEInfo ode_info, int pos=*)
    cdef void add_pre_integration_component(self, SolverComponent c, int pos=*)
