"""
Class implementing a parallel fsf solver.
"""

# local imports
from pysph.parallel.parallel_cell import ParallelCellManager
from pysph.solver.fsf_solver import *


class ParallelFsfSolver(FSFSolver):
    """
    """
    def __init__(self, component_manager=None,
                 kernel=None,
                 integrator=None,
                 time_step=0.0,
                 total_simulation_time=0.0,
                 max_fluid_density_variation=0.01,
                 *args, **kwargs):
        """
        Constructor.
        """
        cell_manager = ParallelCellManager(initialize=False)
        nnps_manager = NNPSManager(cell_manager=cell_manager)

        FSFSolver.__init__(self,
                           cell_manager=cell_manager,
                           nnps_manager=nnps_manager,
                           kernel=kernel,
                           integrator=integrator,
                           time_step=time_step,
                           total_simulation_time=total_simulation_time,
                           max_fluid_density_variation=max_fluid_density_variation)

        # create any new component categories if needed.
        # create the necessary parallel components needed.        

        # following components may have to be added.
        # 
        # (1) A parallel speed of sound computer - to be done once before
        # iterations being.
        #
        # (2) A cell manager initializer - or may be it is does within the cell
        # to cell manager initialize - a new component may not be needed.
        #
        # (3) A load balancer to be added as a pre/post integration
        # component. Another option is to include load balancing with the cell
        # manager updates - this may reduce some communication.
        # 
        # (4) Update bounds/max min values etc, during start or iteration ? or
        # after neighbor updates has been performed.
        
    def _setup_integrator(self):
        """
        Sets up the integrator.

        In addition to the setup done by the FSFSolver, adds some operations
        required for the parallel solver.
        """
        
        FSFSolver._setup_integrator(self)
        
        # now perform parallel specific component addition etc.
        
