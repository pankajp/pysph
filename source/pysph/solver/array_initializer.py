"""
Component to set values of arrays.
"""

# standard imports
import logging
logger = logging.getLogger()

# local imports
from pysph.solver.solver_base import UserDefinedComponent

class ArrayInitializer(UserDefinedComponent):
    """
    Component to set values of arrays in various entities.
    """
    def __init__(self, name='',
                 solver=None,
                 component_manager=None,
                 entity_list=[],
                 array_names=[],
                 array_values=[], *args, **kwargs):
        """
        Constructor.
        """
        UserDefinedComponent.__init__(self, name=name,
                                      solver=solver,
                                      component_manager=component_manager,
                                      entity_list=entity_list)

        if len(array_values) != len(array_names):
            if len(array_values) != 1:
                msg = 'Size of array_names and array_values must be equal'
                msg += '\nOR\n'
                msg += 'Exactly one value must be specified in array_values\n'
                msg += str(array_names)+'\n'
                msg += str(array_values)+'\n'
                
                logger.error(msg)
                raise ValueError, msg
        
        self.array_names = []
        self.array_names[:] = array_names

        self.array_values = []
        if len(array_values) == 1:
            self.array_values[:] = array_values*len(array_names)
        else:
            self.array_values[:] = array_values

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
        The initializer function.
        """
        self.setup_component()

        for entity in self.entity_list:
            parray = entity.get_particle_array()

            if parray is None:
                logger.warn('entity without particle array specified.')
                continue
            
            for i in range(len(self.array_names)):
                a = self.array_names[i]
                val = self.array_values[i]
                arr = parray.get(a)
                arr[:] = val

        return 0
