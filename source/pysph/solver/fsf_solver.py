"""
Generic free surface flow solver.
"""

# standard imports


# local imports


class DensityComputationMode:
    """
    Simple class to specify the method used to compute densities. 
    """
    Simple_Summation = 0
    Continuity_Equation_Update = 1
    def __init__(self, *args, **kwargs):
        """
        Constructor.
        """
        raise SystemError, 'Do not instantiate this class'

class FSFSolver:
    """
    Generic solver for Free Surface Flows.
    """
    def __init__(self):
        """
        Constructor.
        """
        self.component_manager = ComponentManager()
        
        self.density_computation_mode = (
            DensityComputationMode.Continuity_Equation_Update)
        
        self.density_components = []
        self.pressure_components = [TaitPressureComponent()]
        
        self.pressure_gradient_components = [SymmetricPressureGradientComponent()]
        self.viscosity_components = [MonaghanArtViscComponent()]
        self.boundary_force_components = [SPHBoundaryForceComponent()]
        self.density_rate_components = []

        self.time_step_update_components = []

        self.extra_pre_step_components = []
        self.extra_post_step_components = []

        self.pre_integration_components = []
        self.post_integration_components = []

        self.extra_acceleration_components  = []

        self.integrator = RK2XSPHIntegrator()
        
        self.entity_list = []

        self.g = 9.81

        self.total_simulation_time = 1.0
        self.elapsed_time = 0.0
        self.time_step = TimeStep(0.1)
    
    ######################################################################
    # `Public` interface
    ######################################################################
    def add_entity(self, Entity e):
        """
        Adds a physical entity to be included in the simulation.
        """
        pass

    def add_inflow(self, inflow):
        """
        Adds an inflow component to be included in the simulation.
        """
        pass

    def add_outflow(self, outflow):
        """
        Adds an outflow component to be included in the simulation.
        """
        pass

    def add_actuator(self, actuator):
        """
        Adds a component that modifies velocities of some entity in the
        simulation.
        """
        pass

    def solve(self):
        """
        Run the solver.
        """

        # setup the solver for execution.
        self._setup_solver()

        current_time = 0.0
        while current_time < self.total_simulation_time:
            
            self.integrator.integrate()

            # call backs could come here or within the post integration
            # components of the integrator.

            current_time += self.time_step.time_step
            self.elapsed_time = current_time
        
    ######################################################################
    # Non-public interface
    ######################################################################
    def _setup_solver(self):
        """
        Sets up the solver before final interations begin.
        """
        logger.info('Setting up component manager')
        self._setup_component_manager()

        logger.info('Setting up integrator')
        self._setup_integrator()

        logger.info('Setting up entities')
        self._setup_entities()

        logger.info('Setting up components inputs')
        self._setup_component_inputs()

    def _setup_component_manager(self):
        """
        Adds all components to the component manager.
        """
        map(self.component_manager.add_component,
            self.extra_pre_integration_components)
        
        map(self.component_manager.add_component, 
            self.extra_pre_integration_components)
        
        map(self.component_manager.add_component,
            extra_post_integration_components) 
        
        map(self.component_manager.add_component, 
            self.extra_acceleration_components)
        
        map(self.component_manager.add_component, 
            self.exteral_force_components)
        
        map(self.component_manager.add_component, 
            self.density_components)
        
        map(self.component_manager.add_component, 
            self.pressure_components)
        
        map(self.component_manager.add_component, 
            self.pressure_gradient_components)
        
        map(self.component_manager.add_component, 
            self.viscosity_components)
        
        map(self.component_manager.add_component, 
            self.boundary_force_components)
        
    def _setup_component_inputs(self):
        """
        Sets up the inputs of all the components.
        """
        for e in self.entity_list:
            self.component_manager.add_input(e)        

    def _setup_integrator(self):
        """
        Function to setup the integrator before running the solver.

        **Algorithm**

            - if density_computation_mode is Continuity_Equation_Update
                  make sure a component for density update is specified.
                  if not warn and add a default SPH density summation
                  component.
                  add a stepper for density.

            - add all pre-step-integration components to the integrator's pre
              step components.

            - if density_computation_mode is Simple_Summation, add an appropiate
              density component to the pre-step components.

            - add the pressure components to the integrators pre step
              components.

            - add the pressure_gradient, viscosity_components and boundary_force
              components to the velocity steppers pre-step components.
              
            - add a gravity component to the velocity steppers pre-step
              component.

            - add the extra acceleration components to the velocity steppers
              pre-step components.

            - if density_computation_mode is Continuity_Equation_Update add the
              density_rate component to the density steppers pre-step component.

            - add the time_step components to the post-integration
              components. This will be done after all steps of an integrator in
              case a multi-step integrator is chosen.

            - finally add the integrator to the component manager.

        """
        pass

    def _setup_entities(self):
        """
        Function to setup the array requirements of entities.

        **Algorithm**

            - for each entity that has been added using the add_entity function,
            pass them to the setup_entity function of the component manager to
            setup the arrays of the entity.

        """
        pass

