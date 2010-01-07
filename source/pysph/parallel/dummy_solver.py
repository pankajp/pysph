"""
Dummy solver for timing parallel code.
"""

# logging imports
import logging
logger = logging.getLogger()


# local imports
from pysph.base.nnps import *

from pysph.parallel.parallel_cell import ParallelCellManager
from pysph.parallel.parallel_controller import ParallelController
from pysph.parallel.parallel_component import ParallelComponent
from pysph.solver.solver_base import *

class DummySolver(SolverBase):
    def __init__(self, component_manager=None,
                 cell_manager=None,
                 nnps_manager=None,
                 kernel=None,
                 time_step=0.0,
                 total_simulation_time=0.0,
                 parallel_controller =None,
                 *args, **kwargs):
        SolverBase.__init__(self, component_manager=component_manager,
                            cell_manager=cell_manager,
                            nnps_manager=nnps_manager,
                            kernel=kernel,
                            time_step=time_step,
                            total_simulation_time=total_simulation_time)

        if parallel_controller is None:
            self.parallel_controller = ParallelController()
        else:
            self.parallel_controller = parallel_controller
