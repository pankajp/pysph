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
from pysph.solver.density_components import SPHDensityComponent\
    , SPHDensityRateComponent
from pysph.solver.pressure_components import TaitPressureComponent
from pysph.solver.viscosity_components import MonaghanArtViscComponent
from pysph.solver.pressure_gradient_components import \
    SPHSymmetricPressureGradientComponent
from pysph.solver.boundary_force_components import \
    SPHRepulsiveBoundaryForceComponent 
from pysph.solver.integrator_base import Integrator
from pysph.solver.adder_component import AdderComponent
from pysph.solver.xsph_integrator import RK2XSPHIntegrator, EulerXSPHIntegrator
from pysph.solver.runge_kutta_integrator import RK2Integrator
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
                 enable_timing=False,
                 timing_output_file='',
                 max_fluid_density_variation=0.01,
                 *args, **kwargs):
        """
        Constructor.
        """
        SolverBase.__init__(self, component_manager=component_manager,
                            cell_manager=cell_manager,
                            nnps_manager=nnps_manager, 
                            kernel=kernel,
                            integrator=integrator,
                            time_step=time_step,
                            total_simulation_time=total_simulation_time,
                            enable_timing=enable_timing,
                            timing_output_file=timing_output_file,
                            *args, **kwargs)

        logger.debug('FSFSolver Constructor called')

        self.g = -9.81
        self.max_fluid_density_variation = max_fluid_density_variation
        self.speed_of_sound = SpeedOfSound(0.0)

        # setup the various categories of components required.
        self.component_categories['density'] = []
        self.component_categories['pressure'] = []
        self.component_categories['pressure_gradient'] = []
        self.component_categories['viscosity'] = []
        self.component_categories['boundary_force'] = []
        self.component_categories['density_rate'] = []
        #self.component_categories['pre_step'] = []
        #self.component_categories['post_step'] = []
        #self.component_categories['pre_integration'] = []
        #self.component_categories['post_integration'] = []
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

        self.component_categories['viscosity'].append(
            MonaghanArtViscComponent(name='art_visc_default', solver=self))
        
        self.component_categories['boundary_force'].append(
            SPHRepulsiveBoundaryForceComponent(name='boundary_default',
                                               solver=self))

        self.component_categories['density_rate'].append(
            SPHDensityRateComponent(name='density_rate_default',solver=self))

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
            self._component_name_check(c)
            self.integrator.add_pre_integration_component(c.name)

        # add the inflow and outflow componets.
        # both of these are added at the top of the pre-step component as the
        # nnps_updater (added in the base class, should be at the last step of
        # the pre-step components)
        for c in self.component_categories['inflow']:
            self._component_name_check(c)
            self.integrator.add_pre_step_component(c.name, at_tail=False)

        for c in self.component_categories['outflow']:
            self._component_name_check(c)
            self.integrator.add_pre_step_component(c.name, at_tail=False)

        # add the pre-step components
        for c in self.component_categories['pre_step']:
            self._component_name_check(c)
            self.integrator.add_pre_step_component(c.name, at_tail=False)

        # setup the density components.
        self._setup_density_component()

        # add pressure components to the pre-step components
        for c in self.component_categories['pressure']:
            self._component_name_check(c)
            self.integrator.add_pre_step_component(c.name)

        # now add the acceleration computers.
        for c in self.component_categories['pressure_gradient']:
            self._component_name_check(c)
            self.integrator.add_pre_step_component(c.name, 'velocity')

        for c in self.component_categories['viscosity']:
            self._component_name_check(c)
            self.integrator.add_pre_step_component(c.name, 'velocity')
            
        for c in self.component_categories['boundary_force']:
            self._component_name_check(c)
            self.integrator.add_pre_step_component(c.name, 'velocity')

        self._setup_gravity_component()
        
        # add post step components if any.
        for c in self.component_categories['post_step']:
            self._component_name_check(c)
            self.integrator.add_post_step_component(c)

        # add time step updater component.
        for c in self.component_categories['time_step_update']:
            self._component_name_check(c)
            self.integrator.add_post_integration_component(c.name)

        for c in self.component_categories['post_integration']:
            self._component_name_check(c)
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
            
            logger.info('Density Computation : Simple Summation')

            d_list = self.component_categories['density']
            
            if len(d_list) == 0:
                raise SystemError, 'No density components specified'
            else:
                for c in d_list:
                    self.integrator.add_pre_step_component(c.name)
        else:
            logger.info('Density Computation : Continuity Equation')
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
                msg = 'No density rate components specified'
                logger.error(msg)
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
        y_max = -1e20
        particles_present = False

        for e in self.entity_list:
            if e.is_a(EntityTypes.Entity_Fluid):
                particles = e.get_particle_array()
                if particles.get_number_of_particles() == 0:
                    logger.info('No particles found for %s'%(e.name))
                    continue
                y = numpy.max(particles.y)
                particles_present = True
                if y > y_max:
                    y_max = y
        
        if particles_present is False:
            self.speed_of_sound.value = 20.
            logger.info('No particles found, using value of 20.')
        else:
            v = numpy.sqrt(2*9.81*y_max)
            speed = v/numpy.sqrt(self.max_fluid_density_variation)
            self.speed_of_sound.value = speed
            logger.info('Using speed of sound %f'%(speed))
                
