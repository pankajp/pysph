"""
Component to update properties of copies of remote particles in a parallel
simulation. 
"""

# logging imports
import logging
logger = logging.getLogger()

# local imports
from pysph.solver.solver_base import UserDefinedComponent

class ParallelPropertyUpdater(UserDefinedComponent):
    """
    Component to update properties of copies of remote particles in a parallel
    simulation. 
    """
    def __init__(self, 
                 name='',
                 solver=None,
                 component_manager=None,
                 entity_list=[],
                 property_dict={}, *args, **kwargs):
        """
        Constructor.
        """
        UserDefinedComponent.__init__(self, name=name, solver=solver,
                                      component_manager=component_manager,
                                      entity_list=entity_list, *args, **kwargs)

        # this list of properties for each particle array that is being managed
        # by the cell manager.
        self.property_dict = {}
        self.property_dict.update(property_dict)
        
        self.property_list = []

        self.parallel_cell_manager = self.solver.cell_manager
        
    def py_compute(self):
        """
        Update all property values currently. 
        
        We need a clear method to decide which properties are to be updated in
        this function.
        """
        self.parallel_cell_manager.update_remote_particle_properties(None)

        return 0

    def update_property_requirements(self):
        """
        Make up to date any property/array requirements of this component.
        """
        return 0

    def setup_component(self):
        """
        Setup internals of this component.
        """
        return 0    

