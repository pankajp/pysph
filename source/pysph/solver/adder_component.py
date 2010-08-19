"""
Component to add values to arrays.
"""

# standard imports
import logging
logger = logging.getLogger()

# local imports
from pysph.solver.solver_base import UserDefinedComponent

###############################################################################
# `AdderComponent` class.
###############################################################################

# FIXME
#
# 1.Some common operations, like the one in setup_component can be moved into a
# base class.
#
# 2.update_property_requirements should also be implemented so that the array
# requirements are available if this component is added to a component manager.

class AdderComponent(UserDefinedComponent):
    """
    Component to add values to arrays.
    """
    def __init__(self, name='',solver=None, 
                 component_manager=None,
                 entity_list=[],
                 array_names=[],
                 values=[]):
        """
        Constructor.
        """
        if len(array_names) != len(values):
            if len(values) != 1:
                msg = 'Size of array_names and values must be equal'
                msg += '\nOR\n'
                msg += 'Exactly one value must be specified in values\n'
                msg += str(array_names)+'\n'
                msg += str(values)+'\n'
                logger.error(msg)
                raise ValueError, msg

        UserDefinedComponent.__init__(self, name=name, solver=solver,
                                      component_manager=component_manager,
                                      entity_list=entity_list)
        
        self.array_names = []
        self.array_names[:] = array_names
        
        self.values = []
        if len(values) == 1:
            self.values[:] = values*len(self.array_names)
        else:
            self.values[:] = values

    def setup_component(self):
        """
        Sets up the component.
        """
        if self.setup_done == True:
            return

        to_remove = []

        for e in self.entity_list:
            pa = e.get_particle_array()
            if pa is None:
                to_remove.append(e)
                continue
            for a in self.array_names:
                if pa.has_array(a):
                    continue
                else:
                    msg = 'Required properties not present\n'
                    msg += a + '\n'
                    logger.error(msg)
                    raise AttributeError, msg

            # remove entities that were marked for removal.
            for e in to_remove:
                self.entity_list.remove(e)

        self.setup_done = True

    def py_compute(self):
        """
        The adder function.
        """
        self.setup_component()

        for entity in self.entity_list:
            parray = entity.get_particle_array()

            if parray is None:
                logger.warn('entity without particle array specified.')
                continue
            
            for i in range(len(self.array_names)):
                a = self.array_names[i]
                val = self.values[i]
                arr = parray.get(a)
                arr[:] += val

        return 0        
    
