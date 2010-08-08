"""
Class implementing a parallel fsf solver.
"""
# logging imports
import logging
logger = logging.getLogger()

# local imports
from pysph.base.nnps import *

#from pysph.parallel.parallel_cell import ParallelCellManager
from pysph.parallel.parallel_cell import ParallelCellManager
from pysph.parallel.parallel_controller import ParallelController
from pysph.parallel.parallel_component import ParallelComponent
from pysph.solver.parallel_property_updater import ParallelPropertyUpdater
from pysph.solver.fsf_solver import *
from pysph.solver.entity_types import *

################################################################################
# `ParallelFsfSolver` class.
################################################################################
class ParallelFsfSolver(FSFSolver):
    """
    Parallel version of the FSF solver.
    """
    def __init__(self, component_manager=None,
                 cell_manager=None, nnps_manager=None,
                 kernel=None,
                 integrator=None,
                 time_step=0.0,
                 enable_timing=False,
                 timing_output_file='',
                 total_simulation_time=0.0,
                 max_fluid_density_variation=0.01,
                 *args, **kwargs):
        """
        Constructor.

        create any new component categories if needed.
        create the necessary parallel components needed.        

        following components may have to be added.
        
        (1) A parallel speed of sound computer - to be done once before
        iterations being.
        
        (2) A cell manager initializer - or may be it is does within the cell
        to cell manager initialize - a new component may not be needed.
        
        (3) A load balancer to be added as a pre/post integration
        component. Another option is to include load balancing with the cell
        manager updates - this may reduce some communication.
        
        (4) Update bounds/max min values etc, during start or iteration ? or
        after neighbor updates has been performed.

        """
        logger.debug('ParallelFsfSolver, before FSF Constructor')
        parallel_controller = ParallelController()
        cell_manager = ParallelCellManager(
            parallel_controller=parallel_controller, 
            initialize=False,
            solver=self)
                
        FSFSolver.__init__(self,
                           cell_manager=cell_manager, 
                           nnps_manager=nnps_manager,
                           kernel=kernel,
                           integrator=integrator,
                           time_step=time_step,
                           total_simulation_time=total_simulation_time,
                           enable_timing=enable_timing,
                           timing_output_file=timing_output_file,
                           max_fluid_density_variation=max_fluid_density_variation)

        self.parallel_controller = parallel_controller
        self.parallel_controller.solver = self

        # add a parallel component to the component manager.
        pc = ParallelComponent(name='parallel_component', solver=self)
        self.component_manager.add_component(pc)
        
    def _setup_integrator(self):
        """
        Sets up the integrator.

        In addition to the setup done by the FSFSolver, adds some operations
        required for the parallel solver.
        """
        
        FSFSolver._setup_integrator(self)
        
        # now perform parallel specific component addition etc.
        
        # add a property updater as the last component in the pre-step
        # components.
        ppu = ParallelPropertyUpdater(name='parallel_property_updater',
                                      solver=self)
        self.component_manager.add_component(ppu)

        self.integrator.add_pre_step_component(ppu.name, at_tail=True)

    def _setup_solver(self):
        """
        Performs some extra setup in addition to the base class functions.
        """
        SolverBase._setup_solver(self)

        self._setup_speed_of_sound()

    def _setup_speed_of_sound(self):
        """
        Compute the speed of sound.
        """

        y_max = -1e20
        particles_present = False
        for e in self.entity_list:
            if e.is_a(EntityTypes.Entity_Fluid):
                particles = e.get_particle_array()
                if particles.get_number_of_particles() == 0:
                    continue
                y = numpy.max(particles.y)
                particles_present = True
                if y > y_max:
                    y_max = y

        # now find the max y value from all processors.
        pc = self.parallel_controller
        local_max = {'y':y_max}
        local_min = {'y':-1e20}
        
        glb_min, glb_max = pc.get_glb_min_max(local_min, local_max)
        
        y_max = glb_max['y']

        if y_max == -1e20:
            self.speed_of_sound.value = 20.
            logger.info('No particles found, using value of 20.')
        else:
            v = numpy.sqrt(2*9.81*y_max)
            speed = v/numpy.sqrt(self.max_fluid_density_variation)
            self.speed_of_sound.value = speed
            logger.info('Using speed of sound %f'%(speed))        
        
