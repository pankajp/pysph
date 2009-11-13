"""
Module contains classes for performing array copies, moves etc.
"""

# logging import
import logging
logger = logging.getLogger()

# local import 
from pysph.solver.solver_base import UserDefinedComponent

################################################################################
# `ArrayCopier` class.
################################################################################
class ArrayCopier(UserDefinedComponent):
    """
    """
    def __init__(self, name='', solver = None, component_manager=None,
                 entity_list=[], 
                 from_arrays=[], to_arrays=[], 
                 *args, **kwargs):
        """
        Constructor.

        **Parameters**
        
            - name - name of the component.
            - cm   - component manager.
            - entity_list - list of entities on whom to perform operations.
            - from_arrays - list of arrays to copy from.
            - to_arrays - corresponding list of arrays to copy to.
            
        """
        self.entity_list = []
        self.from_arrays = []
        self.to_arrays = []
        
        self.entity_list[:] = entity_list

        # make sure the lengths of the arrays are the same.
        if len(from_arrays) != len(to_arrays):
            raise ValueError, 'From and To array lengths unequal'

        self.from_arrays[:] = from_arrays
        self.to_arrays[:] = to_arrays

    def add_entity(self, entity):
        """
        Adds another entity whose arrays are to be copied.
        """
        if not self.filter_entity(entity):
            self.entity_list.append(entity)

        self.setup_done = False

    def add_array_pair(self, from_array, to_array):
        """
        Add a new (from to) pair.
        
        **Parameters**
        
            - from_array - array to take values from.
            - to_array - array to copy values to.

        """
        self.from_arrays.append(from_array)
        self.to_arrays.append(to_array)

        self.setup_done = False

    def setup_component(self):
        """
        Makes sure the arrays are found in all entities, otherwise raises an
        exception. Entities not providing any particle arrays will be removed.
        """
        to_remove = []
        if not self.setup_done:
            for e in self.entity_list:
                pa = e.get_particle_array()
                if pa is None:
                    to_remove.append(e)
                    continue
                for i in range(len(self.from_arrays)):
                    fa = self.from_arrays[i]
                    ta = self.to_arrays[i]
                    if pa.has_array(fa) and pa.has_array(ta):
                        continue
                    else:
                        raise AttributeError, 'Required properties not present'

            # remove entities that were marked for removal.
            for e in to_remove:
                self.entity_list.remove(e)

            self.setup_done = True
        
    def py_compute(self):
        """
        Copies the contents of arrays in from_arrays to arrays in to_array.

        Note that we implement the py_compute function, as this is a pure python
        module and this class derives from UserDefinedComponent.
        """

        # make sure the component has been properly setup.
        self.setup_component()

        for entity in self.entity_list:
            parray = entity.get_particle_array()
            if parray is None:
                logger.warn('entity without particle array specified.')
                continue
            
            for i in range(len(self.from_arrays)):
                fa_name = self.from_arrays[i]
                ta_name = self.to_arrays[i]
                
                from_arr = parray.get_carray(fa_name).get_npy_array()
                to_arr = parray.get_carray(ta_name).get_npy_array()

                from_arr[:] = to_arr
