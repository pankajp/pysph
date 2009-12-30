"""
A component to be included in all parallel simulations.

This component does not perform any operations. It just adds various particle
attributes required for parallel simulations to the component manager.
"""

# logging imports
import logging
logger = logging.getLogger()

# local imports 
from pysph.solver.solver_base import UserDefinedComponent
from pysph.parallel.parallel_controller import ParallelController
from pysph.solver.entity_types import *

class ParallelComponent(UserDefinedComponent):
    """
    Component class to be included in all parallel simulations.

    This component does not DO anything, hence the setup_component and compute
    functions are empty. We do NOT expect these functions to be called anyways.

    """
    def __init__(self, 
                 name='',
                 solver=None,
                 component_manager=None,
                 entity_list=[],
                 *args, **kwargs):
        """
        Constructor.
        """
        UserDefinedComponent.__init__(
            self, name=name, solver=solver,
            component_manager=component_manager,
            entity_list=entity_list)

        # applies to all entities.
        self.add_input_entity_type(EntityTypes.Entity_Base)
    
    def update_property_requirements(self):
        """
        Update the property requirements.
        """
        # get the pid where is component is running.
        pid = None
        if self.solver is None:
            pc = ParallelController()
            pid = pc.rank
        else:
            pid = self.solver.parallel_controller.rank        

        in_types = self.information.get_dict(self.INPUT_TYPES)

        for t in in_types.keys():
            self.add_write_prop_requirement(t, 'local', 1)
            self.add_write_prop_requirement(t, 'pid', pid)
        
        return 0
