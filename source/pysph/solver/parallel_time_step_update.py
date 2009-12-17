"""
Component to update the time step at all the processes involved in a parallel
simulation.
"""

# logging imports
import logging
logger = logging.getLogger()

# local imports
from pysph.solver.solver_base import *

class ParallelTimeStepUpdateComponent(UserDefinedComponent):
    """
    Component to update the timestep at all processes of  parallel
    simulation. This finds the minimum timestep of all the nodes and uses that
    as the timestep.

    """
    def __init__(self, 
                 name='',
                 solver=None,
                 component_manager=None,
                 time_step_component=None,
                 *args, **kwargs):
        """
        Constructor.
        """
        UserDefinedComponent.__init__(self, name=name, solver=solver,
                                      component_manager=component_manager,
                                      *args, **kwargs)

        self.time_step_component = time_step_component
        self.parallel_controller = self.solver.parallel_controller

    def py_compute(self):
        """
        Find the min time step of all processes and use that as the solvers
        time-step. 
        """
        
        self.setup_component()

        # call the local timestep component
        self.time_step_component.py_compute()

        local_min = {'ts':self.solver.time_step.value}
        local_max = {'ts':self.solver.time_step.value}
        pc = self.parallel_controller
        glb_min, glb_max = pc.get_glb_min_max(local_min, local_max)

        self.solver.time_step.value = glb_min['ts']

        logger.info('Global min timestep : %1.10f'%(self.solver.time_step.value))
        
        return 0

    def update_property_requirements(self):
        """
        """
        return 0

    def setup_component(self):
        """
        """
        if self.setup_done == True:
            return
        
        if self.time_step_component == None:
            msg = 'No time step component set'
            logger.error(msg)
            raise SystemError, msg

        self.setup_done = True

        return 0
