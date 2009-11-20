"""
Generic free surface flow solver.
"""

# standard imports
import numpy
import logging
logger = logging.getLogger()

# local imports
from pysph.solver.solver_base import SolverBase
from pysph.solver.entity_types import EntityTypes
from pysph.solver.density_components import SPHDensityComponent
from pysph.solver.pressure_components import TaitPressureComponent
from pysph.solver.pressure_gradient_components import \
    SPHSymmetricPressureGradientComponent
from pysph.solver.adder_component import AdderComponent
from pysph.solver.xsph_integrator import RK2XSPHIntegrator
from pysph.solver.speed_of_sound import SpeedOfSound


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

class FSFSolver(SolverBase):
    """
    Generic solver for Free Surface Flows.
    """
    def __init__(self, component_manager=None, 
                 cell_manager=None, nnps_manager=None, 
                 kernel=None, 
                 integrator=None,
                 time_step=0.0, 
                 total_simulation_time=0.0,
                 max_fluid_density_variation=0.01,
                 *args, **kwargs):
        """
        Constructor.
        """
        self.g = 9.81
        self.max_fluid_density_variation = max_fluid_density_variation
        self.speed_of_sound = SpeedOfSound(0.0)

        # setup the various categories of components required.
        self.component_categories['density'] = []
        self.component_categories['pressure'] = []
        self.component_categories['pressure_gradient'] = []
        self.component_categories['viscosity'] = []
        self.component_categories['boundary_force'] = []
        self.component_categories['density_rate'] = []
        self.component_categories['pre_step'] = []
        self.component_categories['post_step'] = []
        self.component_categories['pre_integration'] = []
        self.component_categories['post_integration'] = []
        self.component_categories['time_step_update'] = []
        self.component_categories['inflow'] = []
        self.component_categories['outflow'] = []

        self.density_computation_mode = \
            DensityComputationMode.Continuity_Equation_Update

        # setup the default components
        self.component_categories['density'].append(
            SPHDensityComponent(name='density_default', solver=self))

        self.component_categories['pressure'].append(
            TaitPressureComponent(name='pressure_default', solver=self))
        
        self.component_categories['pressure_gradient'].append(
            SPHSymmetricPressureGradientComponent(name='pg_default',
                                                  solver=self))

        #self.component_categories['viscosity'].append(
        #    MonaghanArtViscComponent(name='art_visc_default', solver=self))
        
        #self.component_categories['boundary_force'].append(
        #    SPHBoundaryForceComponent(name='boundary_default', solver=self))

        #self.component_categories['density_rate'].append(
        #    SPHDensityRateComponent(name='density_rate_default',solver=self))

        if self.integrator is None:
            self.integrator = RK2XSPHIntegrator(name='integrator_default',
                                                solver=self)
    
    ######################################################################
    # `Public` interface
    ######################################################################
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
    ######################################################################
    # Non-public interface
    ######################################################################
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
        # add the pre-integration components.
        for c in self.component_categories['pre_integration']:
            self.integrator.add_pre_integration_component(c.name)

        # add the inflow and outflow componets.
        for c in self.component_categories['inflow']:
            self.integrator.add_pre_step_component(c.name)

        for c in self.component_categories['outflow']:
            self.integrator.add_pre_step_component(c.name)

        # add the pre-step components
        for c in self.component_categories['pre_step']:
            self.integrator.add_pre_step_component(c.name)

        # setup the density components.
        self._setup_density_component()

        # add pressure components to the pre-step components
        for c in self.component_categories['pressure']:
            self.integrator.add_pre_step_component(c.name)

        # now add the acceleration computers.
        for c in self.component_categories['pressure_gradient']:
            self.integrator.add_pre_step_component(c.name, 'velocity')

        for c in self.component_categories['viscosity']:
            self.integrator.add_pre_step_component(c.name, 'velocity')
            
        for c in self.component_categories['boundary_force']:
            self.integrator.add_pre_step_component(c.name, 'velocity')

        self._setup_gravity_component()
        
        # add post step components if any.
        for c in self.component_categories['post_step']:
            self.integrator.add_post_step_component(c)

        # add time step updater component.
        for c in self.component_categories['time_step_update']:
            self.integrator.add_post_integration_component(c.name)

    def _setup_gravity_component(self):
        """
        Adds a 'adder' component just prior to the velocity stepper.
        """
        gravity_comp = AdderComponent(name='fsf_gravity_adder',
                                      solver=self,
                                      array_names=['ay'],
                                      values=[self.g])
        
        # make this component accept only fluids as inputs.
        gravity_comp.add_input_entity_type(EntityTypes.Entity_Fluid)

        # add this to the component manager.
        self.component_manager.add_component(gravity_comp, notify=True)

        # add this as a pre-velocity-step component to the integrator.
        self.integrator.add_pre_step_component(gravity_comp.name, 'velocity')
        
    def _setup_density_component(self):
        """
        Depending on the density computation mode, either add a density
        component as pre-step components, or add the rho-rate integration
        property and add density rate components if any.

        """
        if (self.density_computation_mode ==
            DensityComputationMode.Simple_Summation):
            
            d_list = self.component_categories['density']
            
            if len(d_list) == 0:
                raise SystemError, 'No density components specified'
            else:
                for c in d_list:
                    self.integrator.add_pre_step_component(c.name)
        else:
            # if a stepper for density does not exist already
            # add one.
            if self.integrator.get_property_step_info('density') is None:
                # add a stepper for density - this will not be there in the
                # integrator by default.
                i = self.integrator
                i.add_property_step_info(prop_name='density',
                                         integrand_arrays=['rho_rate'],
                                         integral_arrays=['rho'],
                                         entity_types=[EntityTypes.Entity_Fluid],
                                         integrand_initial_values=[0.])

            # now add the density rate components.
            d_list = self.component_categories['density_rate']
    
            if len(d_list) == 0:
                raise SystemError, 'No density rate components specified'
            else:
                for c in d_list:
                    self.integrator.add_pre_step_component(c.name, 'density')

    def _setup_solver(self):
        """
        Performs some extra setup in addition to the base class function.
        """
        
        SolverBase._setup_solver(self)

        self._setup_speed_of_sound()
        
    def _setup_speed_of_sound(self):
        """
        Computes the speed of sound using the fluid particles in the list of
        entities.
        """
        y_max = -1.0

        for e in self.entity_list:
            if e.is_a(EntityTypes.Entity_Fluid):
                particles = e.get_particle_array()
                y = numpy.max(particles.y)
                if y > y_max:
                    y_max = y
        
        v = numpy.sqrt(2*9.81*y_max)
        speed = v/numpy.sqrt(self.max_fluid_density_variation)
        self.speed_of_sound.value = speed
        logger.info('Using speed of sound %f'%(speed))
                
