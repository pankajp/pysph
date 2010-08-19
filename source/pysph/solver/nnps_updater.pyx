"""
Component to update the NNPS.
"""

# logger import
import logging
logger = logging.getLogger()

# local imports
from pysph.base.nnps cimport NNPSManager
from pysph.solver.solver_base cimport SolverComponent, ComponentManager, \
    SolverBase


###############################################################################
# `NNPSUpdater` class.
###############################################################################
cdef class NNPSUpdater(SolverComponent):
    """
    Component to perform nnps updates every iteration.
    """
    category='miscellaneous'
    identifier='nnps_updater'

    def __cinit__(self, SolverBase solver=None,
                  ComponentManager component_manager=None,
                  list entity_list=[],
                  NNPSManager nnps_manager=None,
                  *args, **kwargs):
        """
        Constructor.
        """
        if solver is not None:
            self.nnps_manager = solver.nnps_manager
        else:
            self.nnps_manager = nnps_manager

    cpdef int setup_component(self) except -1:
        """
        """
        if self.nnps_manager is None:
            if self.solver is not None:
                self.nnps_manager = self.solver.nnps_manager

        if self.nnps_manager is None:
            logger.error('nnps manager not set')
            raise ValueError, 'nnps manager not set'

    cdef int compute(self) except -1:
        """
        """
        logger.debug('NNPS UPDATER COMPONENT CALLED')
        self.nnps_manager.update()
        


