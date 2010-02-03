"""
Tests for various classes in the cell.pyx module.
"""

# standard imports
import unittest

# local import
from pysph.base.cell import *
from pysph.base.point import *
from pysph.base.particle_array import ParticleArray
from pysph.base.carray import DoubleArray, LongArray
from pysph.base.tests.common_data import *

def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-16."""
    return numpy.allclose(x, y, atol=1e-16, rtol=0)

class TestModuleFunctions(unittest.TestCase):
    """
    Test various functions in the module.
    """
    def test_real_to_int(self):
        """
        Tests the real_to_int function.
        """
        self.assertEqual(py_real_to_int(0.5, 1.0), 0)
        self.assertEqual(py_real_to_int(0.1, 1.0), 0)

        self.assertEqual(py_real_to_int(1.0, 1.0), 1)        
        self.assertEqual(py_real_to_int(1.5, 1.0), 1)
        self.assertEqual(py_real_to_int(1.9, 1.0), 1)
        
        self.assertEqual(py_real_to_int(2.1, 1.0), 2)
        self.assertEqual(py_real_to_int(2.6, 1.0), 2) 

    def test_find_cell_id(self):
        """
        Tests the find_cell_id function.
        """
        origin = Point(0, 0, 0)
        pnt = Point(0, 0, 1)
        out = IntPoint(0, 0, 0)

        py_find_cell_id(origin, pnt, 1.0, out)
        self.assertEqual(out.x, 0)
        self.assertEqual(out.y, 0)
        self.assertEqual(out.z, 1)

        pnt.x = -2
        py_find_cell_id(origin, pnt, 1.0, out)
        self.assertEqual(out.x, -3)
        self.assertEqual(out.y, 0)
        self.assertEqual(out.z, 1)

        pnt.y = -1
        py_find_cell_id(origin, pnt, 1.0, out)
        self.assertEqual(out.x, -3)
        self.assertEqual(out.y, -2)
        self.assertEqual(out.z, 1)

    def test_find_hierarchy_level_for_radius(self):
        """
        Tests the find_hierarchy_level_for_radius function.
        """
        f = py_find_hierarchy_level_for_radius

        self.assertEqual(f(0.0, 0.1, 0.1, 0.0, 1), 0)
        self.assertEqual(f(0.1, 0.1, 0.1, 0.0, 1), 0)
        self.assertEqual(f(0.11, 0.1, 0.1, 0.0, 1), 1)
        self.assertEqual(f(0.5, 0.1, 0.1, 0.0, 1), 1)
        
        self.assertEqual(f(0.05, 0.1, 0.2, 0.0, 1), 0)
        self.assertEqual(f(0.1, 0.1, 0.2, 0.0, 1), 0)
        self.assertEqual(f(0.11, 0.1, 0.2, 0.0, 1), 1)
        self.assertEqual(f(10.0, 0.1, 0.2, 0.0, 1), 1)

        self.assertEqual(f(0.05, 0.1, 0.2, 0.1, 2), 0)
        self.assertEqual(f(0.1, 0.1, 0.2, 0.1, 2), 0)
        self.assertEqual(f(0.15, 0.1, 0.2, 0.1, 2), 1)
        self.assertEqual(f(0.2, 0.1, 0.2, 0.1, 2), 1)
        self.assertEqual(f(0.21, 0.1, 0.2, 0.1, 2), 2)
        self.assertEqual(f(5.0, 0.1, 0.2, 0.1, 2), 2)

        self.assertEqual(f(0.0, 0.1, 1.1, 0.25, 5), 0)
        self.assertEqual(f(0.1, 0.1, 1.1, 0.25, 5), 0)
        self.assertEqual(f(0.11, 0.1, 1.1, 0.25, 5), 1)
        self.assertEqual(f(0.2, 0.1, 1.1, 0.25, 5), 1)
        self.assertEqual(f(0.3, 0.1, 1.1, 0.25, 5), 1)
        self.assertEqual(f(0.35, 0.1, 1.1, 0.25, 5), 1)
        self.assertEqual(f(0.351, 0.1, 1.1, 0.25, 5), 2)
        self.assertEqual(f(0.45, 0.1, 1.1, 0.25, 5), 2)
        self.assertEqual(f(0.6, 0.1, 1.1, 0.25, 5), 2)
        self.assertEqual(f(0.61, 0.1, 1.1, 0.25, 5), 3)
        self.assertEqual(f(0.79, 0.1, 1.1, 0.25, 5), 3)
        self.assertEqual(f(0.85, 0.1, 1.1, 0.25, 5), 3)
        self.assertEqual(f(0.851, 0.1, 1.1, 0.25, 5), 4)
        self.assertEqual(f(0.951, 0.1, 1.1, 0.25, 5), 4)
        self.assertEqual(f(1.09, 0.1, 1.1, 0.25, 5), 4)

        self.assertEqual(f(1.1, 0.1, 1.1, 0.25, 5), 4)
        self.assertEqual(f(1.11, 0.1, 1.1, 0.25, 5), 5)
        self.assertEqual(f(10.9, 0.1, 1.1, 0.25, 5), 5)

class TestCell(unittest.TestCase):
    """
    Tests for the Cell base class.
    """
    def test_constructor(self):
        cell = Cell(IntPoint(0, 0, 0), cell_manager=None, cell_size=0.1, level=0)
        
        self.assertEqual(cell.id, IntPoint(0, 0, 0))
        self.assertEqual(cell.coord_x=='x', True)
        self.assertEqual(cell.coord_y=='y', True)
        self.assertEqual(cell.coord_z=='z', True)

        self.assertEqual(cell.cell_size == 0.1, True)
        self.assertEqual(cell.level == 0, True)
        self.assertEqual(cell.cell_manager == None, True)
        self.assertEqual(cell.origin == Point(0., 0, 0), True)
        self.assertEqual(cell.jump_tolerance, 1)
        self.assertEqual(cell.arrays_to_bin == [], True)

    def test_set_cell_manager(self):
        """
        Tests the set_cell_manager function.
        """
        cell_manager = CellManager([], None)

        cell = Cell(IntPoint(0, 0, 0), cell_manager=cell_manager, cell_size=0.1, level=0)

        self.assertEqual(cell.arrays_to_bin == [], True)
        self.assertEqual(cell.origin, cell_manager.origin, True)
        self.assertEqual(cell.coord_x, cell_manager.coord_x, True)
        self.assertEqual(cell.coord_y, cell_manager.coord_y, True)
        self.assertEqual(cell.coord_z, cell_manager.coord_z, True)

    def test_get_centroid(self):
        """
        Tests the get_centroid function.
        """
        cell = Cell(IntPoint(0, 0, 0), cell_manager=None, cell_size=0.1, level=0)
        centroid = Point()
        cell.py_get_centroid(centroid)

        self.assertEqual(centroid.x, 0.05)
        self.assertEqual(centroid.y, 0.05)
        self.assertEqual(centroid.z, 0.05)

        cell = Cell(IntPoint(1, 2, 3), cell_manager=None, cell_size=0.5, level=2)
        cell.py_get_centroid(centroid)

        self.assertEqual(centroid.x, 0.75)
        self.assertEqual(centroid.y, 1.25)
        self.assertEqual(centroid.z, 1.75)

################################################################################
# `TestLeafCell` class.
################################################################################     
class TestLeafCell(unittest.TestCase):
    """
    Tests the Leaf cell class.
    """
    def test_constructor(self):
        """
        Tests the leaf cell constructor.
        """
        cell = LeafCell(IntPoint(0, 0, 0), cell_manager=None, cell_size=0.1, level=0)

        self.assertEqual(cell.id, IntPoint(0, 0, 0))
        self.assertEqual(cell.cell_manager, None)
        self.assertEqual(cell.cell_size, 0.1)
        self.assertEqual(cell.level, 0)
        self.assertEqual(cell.jump_tolerance, 1)
        self.assertEqual(cell.index_lists, [])

    def test_set_cell_manager(self):
        """
        Tests the set_cell_manager function.
        """
        cell = LeafCell(IntPoint(0, 0, 0), cell_manager=None, cell_size=0.1, level=0)

        p_arrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=p_arrs, initialize=False)
        cell.set_cell_manager(cm)

        self.assertEqual(len(cell.index_lists), 2)
        
    def test_update(self):
        """
        Tests the update function.
        """
        p_arrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=p_arrs, initialize=False)
        
        cell = LeafCell(IntPoint(0, 0, 0), cell_manager=cm, cell_size=1.0, level=0, jump_tolerance=1)

        # put all the particles into this cell.
        indx_arr = cell.index_lists[0]
        indx_arr.resize(8)
        indx_arr.set_data(numpy.arange(8, dtype=numpy.int))

        indx_arr = cell.index_lists[1]
        indx_arr.resize(4)
        indx_arr.set_data(numpy.arange(4, dtype=numpy.int))
        
        data = {}

        cell.py_update(data)

        # data should contain 6 cells
        self.assertEqual(len(data.values()), 6)

        self.assertEqual(data.has_key(IntPoint(1, 1, 0)), True)
        self.assertEqual(data.has_key(IntPoint(1, 0, 0)), True)
        self.assertEqual(data.has_key(IntPoint(0, 1, 0)), True)
        self.assertEqual(data.has_key(IntPoint(0, 0, 1)), True)
        self.assertEqual(data.has_key(IntPoint(1, 0, 1)), True)
        self.assertEqual(data.has_key(IntPoint(1, 0, -1)), True)
            
        cell_0_1_0 = data[IntPoint(0, 1, 0)]
        # cell_0_1 should contain point 5
        ind_arr = cell_0_1_0.index_lists[0]
        self.assertEqual(ind_arr.length, 1)
        self.assertEqual(ind_arr.get(0), 5)
        ind_arr = cell_0_1_0.index_lists[1]
        self.assertEqual(ind_arr.length, 0)

        
        cell_1_0_0 = data[IntPoint(1, 0, 0)]
        ind_arr = cell_1_0_0.index_lists[0]
        self.assertEqual(ind_arr.length, 1)
        self.assertEqual(ind_arr.get(0), 6)
        ind_arr = cell_1_0_0.index_lists[1]
        self.assertEqual(ind_arr.length, 0)


        cell_1_1_0 = data[IntPoint(1, 1, 0)]
        ind_arr = cell_1_1_0.index_lists[0]
        self.assertEqual(ind_arr.length, 1)
        self.assertEqual(ind_arr.get(0), 7)
        ind_arr = cell_1_1_0.index_lists[1]
        self.assertEqual(ind_arr.length, 0)


        cell_0_0_1 = data[IntPoint(0, 0, 1)]
        ind_arr = cell_0_0_1.index_lists[0]
        self.assertEqual(ind_arr.length, 0)
        ind_arr = cell_0_0_1.index_lists[1]
        self.assertEqual(ind_arr.length, 1)
        self.assertEqual(ind_arr.get(0), 0)

        cell_1_0_1 = data[IntPoint(1, 0, 1)]
        ind_arr = cell_1_0_1.index_lists[0]
        self.assertEqual(ind_arr.length, 0)
        ind_arr = cell_1_0_1.index_lists[1]
        self.assertEqual(ind_arr.length, 1)
        self.assertEqual(ind_arr.get(0), 1)

        cell_1_0__1 = data[IntPoint(1, 0, -1)]
        ind_arr = cell_1_0__1.index_lists[0]
        self.assertEqual(ind_arr.length, 0)
        ind_arr = cell_1_0__1.index_lists[1]
        self.assertEqual(ind_arr.get(0), 2)
        
        # check cell also
        ind_arr = cell.index_lists[0]
        self.assertEqual(ind_arr.length, 5)
        arr = ind_arr.get_npy_array().copy()
        arr.sort()
        self.assertEqual(arr[0], 0)
        self.assertEqual(arr[1], 1)
        self.assertEqual(arr[2], 2)
        self.assertEqual(arr[3], 3)
        self.assertEqual(arr[4], 4)

        ind_arr = cell.index_lists[1]
        self.assertEqual(ind_arr.length, 1)
        self.assertEqual(ind_arr.get(0), 3)

    def test_get_particle_counts_ids(self):
        """
        Tests the get_particle_counts_ids function.
        """
        p_arrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=p_arrs, initialize=False)
        
        cell = LeafCell(IntPoint(0, 0, 0), cell_manager=cm, cell_size=1.0,
                        level=0, jump_tolerance=1)
        
        # put all the particles into this cell.
        indx_arr = cell.index_lists[0]
        indx_arr.resize(8)
        indx_arr.set_data(numpy.arange(8, dtype=numpy.int))

        indx_arr = cell.index_lists[1]
        indx_arr.resize(4)
        indx_arr.set_data(numpy.arange(4, dtype=numpy.int))

        index_lists = []
        counts = LongArray()

        cell.get_particle_counts_ids(index_lists, counts)

        self.assertEqual(len(index_lists), 2)
        self.assertEqual(index_lists[0].length, 8)
        self.assertEqual(index_lists[1].length, 4)

        self.assertEqual(counts[0], 8)
        self.assertEqual(counts[1], 4)

    def test_add_particles(self):
        """
        Tests the add_particles function.
        """
        p_arrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=p_arrs, initialize=False)
        
        cell1 = LeafCell(IntPoint(0, 0, 0), cell_manager=cm, cell_size=1.0, level=0, jump_tolerance=1)
        cell2 = LeafCell(IntPoint(0, 0, 0), cell_manager=cm, cell_size=1.0, level=0, jump_tolerance=1)
        cell3 = LeafCell(IntPoint(0, 0, 0), cell_manager=cm, cell_size=1.0, level=0, jump_tolerance=1)
        
        indx_arr = cell1.index_lists[0]
        indx_arr.resize(2)
        indx_arr[0] = 0
        indx_arr[1] = 1
        
        indx_arr = cell2.index_lists[0]
        indx_arr.resize(3)
        indx_arr[0] = 2
        indx_arr[1] = 3
        indx_arr[2] = 4

        indx_arr = cell3.index_lists[1]
        indx_arr.resize(1)
        indx_arr[0] = 2

        cell1.py_add_particles(cell2)

        # make sure that cell1 now contains indices from cell2 also
        indx_arr = cell1.index_lists[0]
        check_array(indx_arr.get_npy_array(), [0, 1, 2, 3, 4])

        cell1.py_add_particles(cell3)
        indx_arr = cell1.index_lists[1]
        check_array(indx_arr.get_npy_array(), [0, 2])

        # now try adding an index that does not exist, a RuntimeError should be raised.
        indx_arr = cell1.index_lists[0]
        indx_arr.resize(2)
        indx_arr[0] = 0
        indx_arr[1] = 1
        
        indx_arr = cell2.index_lists[0]
        indx_arr.resize(3)
        indx_arr[0] = 2
        indx_arr[1] = 3
        indx_arr[2] = 10

        indx_arr = cell3.index_lists[0]
        indx_arr.resize(1)
        indx_arr[0] = -1
        
        self.assertRaises(RuntimeError, cell1.py_add_particles, cell2)
        self.assertRaises(RuntimeError, cell1.py_add_particles, cell3)
    
    def test_update_cell_manager_hierarchy_list(self):
        """
        Tests the update_cell_manager_hierarchy_list function.
        """
        p_arrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=p_arrs, initialize=False)
        
        cell1 = LeafCell(IntPoint(0, 0, 0), cell_manager=cm, cell_size=1.0, level=0, jump_tolerance=1)
        cell2 = LeafCell(IntPoint(0, 1, 0), cell_manager=cm, cell_size=1.0, level=0, jump_tolerance=1)
        
        cm.py_setup_hierarchy_list()

        cell1.py_update_cell_manager_hierarchy_list()
        cell2.py_update_cell_manager_hierarchy_list()

        self.assertEqual(cm.hierarchy_list[0][IntPoint(0, 0, 0)], cell1)
        self.assertEqual(cm.hierarchy_list[0][IntPoint(0, 1, 0)], cell2)
                
################################################################################
# `TestNonLeafCell` class.
################################################################################
class TestNonLeafCell(unittest.TestCase):
    """
    Tests the NonLeafCell class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        cell = NonLeafCell(IntPoint(0, 0, 0), cell_manager=None, cell_size=0.1, level=1)

        self.assertEqual(cell.id, IntPoint(0, 0, 0))
        self.assertEqual(cell.cell_manager, None)
        self.assertEqual(cell.cell_size, 0.1)
        self.assertEqual(cell.level, 1)
        self.assertEqual(cell.cell_dict, {})

    def test_add_particles(self):
        """
        Tests the add_particles function.
        """
        p_arrs = generate_sample_dataset_2()
        cm = CellManager(arrays_to_bin=p_arrs, initialize=False)
        
        big_cell_source = NonLeafCell(IntPoint(0, 0, 0), cell_manager=cm,
                               cell_size=2.0, level=1)
        small_cell1 = LeafCell(IntPoint(0, 0, 0), cell_manager=cm,
                               cell_size=1.0, level=0)
        indx_arr = small_cell1.index_lists[0]
        indx_arr.resize(1)
        indx_arr[0] = 3
        
        small_cell2 = LeafCell(IntPoint(1, 0, 0), cell_manager=cm,
                               cell_size=1.0, level=0)
        indx_arr = small_cell2.index_lists[0]
        indx_arr.resize(1)
        indx_arr[0] = 4

        big_cell_source.cell_dict[small_cell2.id.py_copy()] = small_cell2
        big_cell_source.cell_dict[small_cell1.id.py_copy()] = small_cell1
        
        big_cell_dest = NonLeafCell(IntPoint(0, 0, 0), cell_manager=cm,
                               cell_size=2.0, level=1)
        
        small_cell_3 = LeafCell(IntPoint(0, 1, 0), cell_manager=cm,
                                cell_size=1.0, level=0)

        
        indx_arr = small_cell_3.index_lists[0]
        indx_arr.resize(1)
        indx_arr[0] = 2

        big_cell_dest.cell_dict[small_cell_3.id.py_copy()] = small_cell_3

        big_cell_dest.py_add_particles(big_cell_source)
        
        # make sure that big_cell_dest has 3 particles now.
        self.assertEqual(big_cell_dest.py_get_number_of_particles(), 3)
        self.assertEqual(len(big_cell_dest.cell_dict), 3)
        
        self.assertEqual(big_cell_dest.cell_dict.has_key(IntPoint(0, 0, 0)),
                         True)
        self.assertEqual(big_cell_dest.cell_dict.has_key(IntPoint(1, 0, 0)), 
                         True)
        self.assertEqual(big_cell_dest.cell_dict.has_key(IntPoint(0, 1, 0)),
                         True)
        small_cell = big_cell_dest.cell_dict[IntPoint(0, 0, 0)]
        self.assertEqual(small_cell.py_get_number_of_particles(), 1)

        small_cell = big_cell_dest.cell_dict[IntPoint(0, 1, 0)]
        self.assertEqual(small_cell.py_get_number_of_particles(), 1)

        small_cell = big_cell_dest.cell_dict[IntPoint(1, 0, 0)]
        self.assertEqual(small_cell.py_get_number_of_particles(), 1)
        
    def test_update(self):
        """
        Tests the update function.
        """
        p_arrs = generate_sample_dataset_2()
        cm = CellManager(arrays_to_bin=p_arrs, initialize=False)
        
        big_cell = NonLeafCell(IntPoint(0, 0, 0), cell_manager=cm,
                               cell_size=2.0, level=1)
        small_cell_1 = LeafCell(IntPoint(0, 0, 0), cell_manager=cm,
                                cell_size=1.0, level=0, jump_tolerance=2)
                
        big_cell.cell_dict[small_cell_1.id.py_copy()] = small_cell_1

        # add all particle indices to small_cell_1
        indx_arr = small_cell_1.index_lists[0]
        indx_arr.resize(7)
        indx_arr.set_data(numpy.arange(7, dtype=numpy.int))

        data = {}

        big_cell.py_update(data)

        self.assertEqual(len(data), 4)
        self.assertEqual(data.has_key(IntPoint(1, 0, 0)), True)
        self.assertEqual(data.has_key(IntPoint(1, -1, 0)), True)
        self.assertEqual(data.has_key(IntPoint(-1, -1, 0)), True)
        self.assertEqual(data.has_key(IntPoint(-1, 1, 0)), True)
        
        # now make sure the cells have the correct data.
        cell_m1_1_0 = data[IntPoint(-1, 1, 0)]
        self.assertEqual(cell_m1_1_0.py_get_number_of_particles(), 1)
        self.assertEqual(len(cell_m1_1_0.cell_dict), 1)
        self.assertEqual(cell_m1_1_0.cell_dict.has_key(IntPoint(-1, 2, 0)),
                         True)
        cell_m1_1_0_child = cell_m1_1_0.cell_dict[IntPoint(-1, 2, 0)]
        indx_arr = cell_m1_1_0_child.index_lists[0]
        self.assertEqual(indx_arr.length, 1)
        self.assertEqual(indx_arr.get(0), 0)

        cell_m1_m1_0 = data[IntPoint(-1, -1, 0)]
        self.assertEqual(cell_m1_m1_0.py_get_number_of_particles(), 1)
        self.assertEqual(len(cell_m1_m1_0.cell_dict), 1)
        self.assertEqual(cell_m1_m1_0.cell_dict.has_key(IntPoint(-1, -1, 0)),
                         True)
        cell_m1_m1_0_child = cell_m1_m1_0.cell_dict[IntPoint(-1, -1, 0)]
        indx_arr = cell_m1_m1_0_child.index_lists[0]
        self.assertEqual(indx_arr.length, 1)
        self.assertEqual(indx_arr.get(0), 1)

        cell_1_0_0 = data[IntPoint(1, 0, 0)]
        self.assertEqual(cell_1_0_0.py_get_number_of_particles(), 1)
        self.assertEqual(len(cell_1_0_0.cell_dict), 1)
        self.assertEqual(cell_1_0_0.cell_dict.has_key(IntPoint(2, 0, 0)), True)
        
        cell_1_0_0_child = cell_1_0_0.cell_dict[IntPoint(2, 0, 0)]
        indx_arr = cell_1_0_0_child.index_lists[0]
        self.assertEqual(indx_arr.length, 1)
        self.assertEqual(indx_arr.get(0), 5)

        cell_1_m1_0 = data[IntPoint(1, -1, 0)]
        self.assertEqual(cell_1_m1_0.py_get_number_of_particles(), 1)
        self.assertEqual(len(cell_1_m1_0.cell_dict), 1)
        self.assertEqual(cell_1_m1_0.cell_dict.has_key(IntPoint(2, -1, 0)), True)
        
        cell_1_m1_0_child = cell_1_m1_0.cell_dict[IntPoint(2, -1, 0)]
        indx_arr = cell_1_m1_0_child.index_lists[0]
        self.assertEqual(indx_arr.length, 1)
        self.assertEqual(indx_arr.get(0), 6)

        # now make sure the 'big_cell' has proper data.
        self.assertEqual(len(big_cell.cell_dict), 3)
        self.assertEqual(big_cell.py_get_number_of_particles(), 3)

        # make sure the proper cells are there
        # we do NOT check if the data in the smaller cells are proper
        # as that should be checked by the test_update function for
        # the smaller cell.
        self.assertEqual(big_cell.cell_dict.has_key(IntPoint(0, 0, 0)), True)
        self.assertEqual(big_cell.cell_dict.has_key(IntPoint(0, 1, 0)), True)
        self.assertEqual(big_cell.cell_dict.has_key(IntPoint(1, 0, 0)), True)                           

    def test_update_cell_manager_hierarchy_list(self):
        """
        Tests the update_cell_manager_hierarchy_list function.
        """
        p_arrs = generate_sample_dataset_2()
        cm = CellManager(arrays_to_bin=p_arrs, initialize=False)
        cm.py_setup_hierarchy_list()
        
        big_cell = NonLeafCell(IntPoint(0, 0, 0), cell_manager=cm,
                               cell_size=2.0, level=1)
        small_cell1 = LeafCell(IntPoint(0, 0, 0), cell_manager=cm,
                               cell_size=1.0, level=0)
        
        small_cell2 = LeafCell(IntPoint(1, 0, 0), cell_manager=cm,
                               cell_size=1.0, level=0)

        big_cell.cell_dict[small_cell2.id.py_copy()] = small_cell2
        big_cell.cell_dict[small_cell1.id.py_copy()] = small_cell1
        
        big_cell.py_update_cell_manager_hierarchy_list()        

        # make sure all entries have been made in the hierarchy_list
        self.assertEqual(len(cm.hierarchy_list[0]), 2)
        self.assertEqual(len(cm.hierarchy_list[1]), 1)

        self.assertEqual(cm.hierarchy_list[0][IntPoint(0, 0, 0)], small_cell1)
        self.assertEqual(cm.hierarchy_list[0][IntPoint(1, 0, 0)], small_cell2)

        self.assertEqual(cm.hierarchy_list[1][IntPoint(0, 0, 0)], big_cell)

    def test_delete_empty_cells(self):
        """
        Tests the delete_empty_cells function.
        """
        p_arrs = generate_sample_dataset_2()
        cm = CellManager(arrays_to_bin=p_arrs, initialize=False)
        cm.py_setup_hierarchy_list()
        
        # create a big cell with one particle in one small cell and
        # one empty name cell.
        big_cell_1 = NonLeafCell(IntPoint(0, 0, 0), cell_manager=cm,
                               cell_size=2.0, level=1)
        # add one particle index to small_cell1
        small_cell1 = LeafCell(IntPoint(0, 0, 0), cell_manager=cm,
                               cell_size=1.0, level=0)
        indx_arr = small_cell1.index_lists[0]
        indx_arr.resize(1)

        small_cell2 = LeafCell(IntPoint(1, 0, 0), cell_manager=cm,
                               cell_size=1.0, level=0)

        big_cell_1.cell_dict[small_cell2.id.py_copy()] = small_cell2
        big_cell_1.cell_dict[small_cell1.id.py_copy()] = small_cell1
        
        # create another big cell with no particles.
        big_cell_2 = NonLeafCell(IntPoint(0, 1, 0), cell_manager=cm,
                                 cell_size=2.0, level=1)

        # create a bigger cell with these cells inside.
        bigger_cell = NonLeafCell(IntPoint(0, 0, 0), cell_manager=cm,
                                  cell_size=4.0, level=2)

        bigger_cell.cell_dict[IntPoint(0, 0, 0)] = big_cell_1
        bigger_cell.cell_dict[IntPoint(0, 1, 0)] = big_cell_2
        
        bigger_cell.py_delete_empty_cells()

        self.assertEqual(len(bigger_cell.cell_dict), 1)
        self.assertEqual(bigger_cell.cell_dict.has_key(IntPoint(0, 0, 0)), True)

        small_cell1 = bigger_cell.cell_dict[IntPoint(0, 0, 0)]
        self.assertEqual(len(small_cell1.cell_dict), 1)
        self.assertEqual(small_cell1.cell_dict.has_key(IntPoint(0, 0, 0)), True)        

################################################################################
# `TestRootCell` class.
################################################################################
class TestRootCell(unittest.TestCase):
    """
    Tests the root cell class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        cell = RootCell()
        self.assertEqual(cell.id, IntPoint(0, 0, 0))
        self.assertEqual(cell.cell_manager, None)
        self.assertEqual(cell.level, -1)
        self.assertEqual(cell.cell_size, 0.1)

    def test_update(self):
        """
        Tests the update function.
        """
        p_arrs = generate_sample_dataset_2()
        cm = CellManager(arrays_to_bin=p_arrs, initialize=False)

        root_cell = RootCell(cell_manager=cm)

        big_cell = NonLeafCell(IntPoint(0, 0, 0), cell_manager=cm,
                               cell_size=2.0, level=1)
        small_cell_1 = LeafCell(IntPoint(0, 0, 0), cell_manager=cm,
                                cell_size=1.0, level=0, jump_tolerance=2)
                
        big_cell.cell_dict[small_cell_1.id.py_copy()] = small_cell_1
        root_cell.cell_dict[big_cell.id.py_copy()] = big_cell

        # add all particle indices to small_cell_1
        indx_arr = small_cell_1.index_lists[0]
        indx_arr.resize(7)
        indx_arr.set_data(numpy.arange(7, dtype=numpy.int))

        root_cell.py_update(None)

        # now root cell should have a total of 5 cells.
        self.assertEqual(len(root_cell.cell_dict), 5)

        self.assertEqual(root_cell.cell_dict.has_key(IntPoint(-1, 1, 0)), True)
        self.assertEqual(root_cell.cell_dict.has_key(IntPoint(-1, -1, 0)), True)
        self.assertEqual(root_cell.cell_dict.has_key(IntPoint(1, 0, 0)), True)
        self.assertEqual(root_cell.cell_dict.has_key(IntPoint(1, -1, 0)), True)
        self.assertEqual(root_cell.cell_dict.has_key(IntPoint(0, 0, 0)), True)

        # we do not test furhter as those test would have been performed by
        # tests for NonLeafCell.                

################################################################################
# `TestCellManager` class.
################################################################################
class TestCellManager(unittest.TestCase):
    """
    Tests for the CellManager class.
    """
    def generate_random_particle_data(self, num_arrays, num_particles):
        """
        Returns a list of particle arrays with random data.
        """
        ret = []
        name = ''
       
        for i in range(num_arrays):
            x = numpy.random.rand(num_particles)
            y = numpy.random.rand(num_particles)
            z = numpy.random.rand(num_particles)
            name = 'arr' + str(i)

            p_arr = ParticleArray(None, name, **{'x':{'data':x}, 'y':{'data':y}, 'z':{'data':z}})
            ret.append(p_arr)
        
        return ret

    def test_constructor(self):
        """
        Tests the constructor.
        """
        cm = CellManager(initialize=False)
        
        # Some checks that should hold prior to cell_manager initialization.
        self.assertEqual(cm.particle_manager, None)
        self.assertEqual(cm.origin, Point(0, 0, 0))
        self.assertEqual(cm.num_levels, 1)
        self.assertEqual(cm.cell_sizes.length, 0)
        self.assertEqual(len(cm.array_indices), 0)
        self.assertEqual(len(cm.arrays_to_bin), 0)
        self.assertEqual(cm.min_cell_size, 0.1)
        self.assertEqual(cm.max_cell_size, 0.5)
        self.assertEqual(cm.jump_tolerance, 1)
        self.assertEqual(cm.coord_x, 'x')
        self.assertEqual(cm.coord_y, 'y')
        self.assertEqual(cm.coord_z, 'z')
        self.assertEqual(len(cm.hierarchy_list), 0)
        
        # now call initialize
        cm.py_initialize()

        # the root cell should be in the top of the hierarchy
        # which means the cm.num_level'th entry.
        self.assertEqual(len(cm.hierarchy_list[cm.num_levels]), 1)
        self.assertEqual(cm.hierarchy_list[cm.num_levels].values()[0],
                         cm.root_cell) 
        
        # there should be no cell in the leaf level.
        self.assertEqual(len(cm.hierarchy_list[0]), 0)
        self.assertEqual(len(cm.hierarchy_list[1]), 1)
        self.assertEqual(cm.hierarchy_list[1].values()[0], cm.root_cell)
        self.assertEqual(cm.root_cell.py_get_number_of_particles(), 0)

    def test_rebuild_array_indices(self):
        """
        Tests the _rebuild_array_indices function.
        """
        p_arrs = self.generate_random_particle_data(3, 10)
        cm = CellManager(arrays_to_bin=p_arrs, initialize=False)

        cm.py_rebuild_array_indices()

        # make sure the array_indices member is set properly.
        self.assertEqual(len(cm.arrays_to_bin), 3)
        self.assertEqual(len(cm.array_indices), 3)

        self.assertEqual(cm.array_indices[p_arrs[0].name], 0)
        self.assertEqual(cm.array_indices[p_arrs[1].name], 1)
        self.assertEqual(cm.array_indices[p_arrs[2].name], 2)

    def test_setup_hierarchy_list(self):
        """
        Tests the _setup_hierarchy_list function.
        """
        p_arrs = self.generate_random_particle_data(3, 10)
        cm = CellManager(arrays_to_bin=p_arrs, initialize=False)

        cm.py_setup_hierarchy_list()

        self.assertEqual(len(cm.hierarchy_list), cm.num_levels+1)
        for d in cm.hierarchy_list:
            self.assertEqual(len(d.items()), 0)
                           
    def test_compute_cell_sizes(self):
        """
        Tests the compute_cell_sizes function.
        """
        cm = CellManager(initialize=False)
        
        cm.py_compute_cell_sizes(cm.min_cell_size, cm.max_cell_size,
                                 cm.num_levels, cm.cell_sizes) 

        self.assertEqual(cm.cell_sizes.length, 1)
        self.assertEqual(cm.cell_sizes[0], cm.min_cell_size)

        # trying out some more calls to compute_cell_sizes
        arr = DoubleArray(10)
        
        cm.py_compute_cell_sizes(1, 10, 10,  arr)

        self.assertEqual(check_array(arr.get_npy_array(), 
                                     numpy.arange(1, 11, dtype=float)), True)

    def test_build_base_hierarchy(self):
        """
        Tests the _build_base_hierarchy function.
        """
        p_arrs = self.generate_random_particle_data(3, 10)
        cm = CellManager(arrays_to_bin=p_arrs, initialize=False)

        cm.py_rebuild_array_indices()
        cm.py_setup_hierarchy_list()
        cm.py_build_base_hierarchy()

        # we expect one cell at each level of the hierarchy.
        # the lowest level cell should have all the particles.
        self.assertEqual(len(cm.hierarchy_list), 2)
        self.assertEqual(len(cm.hierarchy_list[0]), 1)
        self.assertEqual(len(cm.hierarchy_list[1]), 1)

        # the leaf cell should be the only child of the root cell
        leaf_cell = cm.hierarchy_list[0].values()[0]
        root_cell = cm.hierarchy_list[1].values()[0]
        self.assertEqual(root_cell.cell_dict[leaf_cell.id], leaf_cell)

        # the leaf cell should have all the 30 particle indices
        self.assertEqual(leaf_cell.py_get_number_of_particles(), 30)
        self.assertEqual(root_cell.py_get_number_of_particles(), 30)

        # make sure the jump tolerance of the leaf cells are set to max
        self.assertEqual(leaf_cell.jump_tolerance, INT_INF())

        p_arrs = self.generate_random_particle_data(3, 10)
        cm = CellManager(arrays_to_bin=p_arrs, initialize=False, num_levels=2)

        cm.py_rebuild_array_indices()
        cm.py_setup_hierarchy_list()
        cm.py_build_base_hierarchy()

        for i in range(3):
            self.assertEqual(len(cm.hierarchy_list[i]), 1)
        
        for i in range(2):
            c = cm.hierarchy_list[i].values()[0]
            n = cm.hierarchy_list[i+1].values()[0]
            self.assertEqual(n.cell_dict.values()[0], c)
        
    def test_data_set_1(self):
        """
        Test initialize with data from generate_sample_dataset_1.
        """
        p_arrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=p_arrs, min_cell_size=1.,
                         max_cell_size=2.) 
        
        # there should be two levels in the hierarchy
        self.assertEqual(len(cm.hierarchy_list), 2)
        root_cell = cm.root_cell
        self.assertEqual(cm.hierarchy_list[1].values()[0], root_cell)
        self.assertEqual(len(cm.hierarchy_list[0]), 7)

    def test_data_set_2(self):
        """
        Test initialize with data from generate_sample_dataset_2
        """
        p_arrs = generate_sample_dataset_2()
        cm = CellManager(arrays_to_bin=p_arrs, min_cell_size=1.,
                         max_cell_size=2., num_levels=2)

        self.assertEqual(len(cm.hierarchy_list), 3)
        root_cell = cm.root_cell
        self.assertEqual(len(root_cell.cell_dict), 5)
        self.assertEqual(root_cell.py_get_number_of_particles(), 7)

        # further checking is not needed, the update test of the RootCell
        # would have handled that.

    def test_update(self):
        """
        Tests the update function.

        Use the dataset generated by generate_sample_dataset_2.

        We move two particles,  num 3, num 5 and num 7.
        """
        p_arrs = generate_sample_dataset_2()
        cm = CellManager(arrays_to_bin=p_arrs, min_cell_size=1.,
                         max_cell_size=2., num_levels=2)
        
        # move particle 3 in +ve y direction by one unit.
        y = p_arrs[0].get_carray('y')
        curr_y_2 = y.get(2)
        y[2] = curr_y_2 + 1.0
        x = p_arrs[0].get_carray('x')
        curr_x_6 = x.get(6)
        x[6] = curr_x_6 - 1.0
        curr_y_4 = y.get(4)
        y[4] = curr_y_4 - 1.0

        self.assertEqual(cm.root_cell.py_get_number_of_particles(), 7)
        
        # because we chagned the data, ask cm to update_status
        cm.py_update_status()
        cm.py_update()

        # no change should occur, as we have NOT set the dirty bit of the
        # parray.
        self.assertEqual(len(cm.root_cell.cell_dict), 5)
        root_cell = cm.root_cell

        self.assertEqual(cm.root_cell.py_get_number_of_particles(), 7)
        
        self.assertEqual(root_cell.cell_dict.has_key(IntPoint(0, 0, 0)), True)
        self.assertEqual(root_cell.cell_dict.has_key(IntPoint(1, 0, 0)), True)
        self.assertEqual(root_cell.cell_dict.has_key(IntPoint(1, -1, 0)), True)
        self.assertEqual(root_cell.cell_dict.has_key(IntPoint(-1, -1, 0)), True)
        self.assertEqual(root_cell.cell_dict.has_key(IntPoint(-1, 1, 0)), True)

        cell = root_cell.cell_dict[IntPoint(0, 0, 0)]
        self.assertEqual(cell.py_get_number_of_particles(), 3)
        cell = root_cell.cell_dict[IntPoint(1, 0, 0)]
        self.assertEqual(cell.py_get_number_of_particles(), 1)
        cell = root_cell.cell_dict[IntPoint(1, -1, 0)]
        self.assertEqual(cell.py_get_number_of_particles(), 1)
        cell = root_cell.cell_dict[IntPoint(-1, -1, 0)]
        self.assertEqual(cell.py_get_number_of_particles(), 1)
        cell = root_cell.cell_dict[IntPoint(-1, 1, 0)]
        self.assertEqual(cell.py_get_number_of_particles(), 1)

        # now set it to dirty and check.
        p_arrs[0].set_dirty(True)

        cm.py_update_status()
        cm.py_update()

        self.assertEqual(len(root_cell.cell_dict), 6)
        self.assertEqual(root_cell.py_get_number_of_particles(), 7)

        # make sure the proper cells have been created.
        cell = root_cell.cell_dict[IntPoint(0, 0, 0)]
        self.assertEqual(len(cell.cell_dict), 1)
        self.assertEqual(cell.cell_dict.has_key(IntPoint(0, 0, 0)), True)

        # cell 1, -1, 0 at level 1 should have been deleted.
        self.assertEqual(root_cell.cell_dict.has_key(IntPoint(1, -1, 0)), False)

        # two new cells should have been created.
        self.assertEqual(root_cell.cell_dict.has_key(IntPoint(0, -1, 0)), True)
        self.assertEqual(root_cell.cell_dict.has_key(IntPoint(0, 1, 0)), True)

        cell = root_cell.cell_dict[IntPoint(0, -1, 0)]
        self.assertEqual(len(cell.cell_dict), 1)
        self.assertEqual(cell.cell_dict.has_key(IntPoint(1, -1, 0)), True)
        self.assertEqual(cell.py_get_number_of_particles(), 2)
        

        cell = root_cell.cell_dict[IntPoint(0, 1, 0)]
        self.assertEqual(len(cell.cell_dict), 1)
        self.assertEqual(cell.py_get_number_of_particles(), 1)
        self.assertEqual(cell.cell_dict.has_key(IntPoint(0, 2, 0)), True)

    def test_get_potential_cells_1(self):
        """
        Tests the get_potential_cells function.
        
        Data used is from generate_sample_dataset_2.

        Search facilities that will be needed:
         
         - given point and interaction radius, find all cells, that are
           immediate neighbors to the cell containing the given point.
         - given point and interaction radius, find all cells, that are possibly
           within its interaction radius.
         - the cells returned in either of the above cases should be at the
           correct level in the hierarchy. 
         - All cells returned will be in the same level in the hierarchy. This
           can be changed later.
         - **All particles are assumed to have constant interaction radius**.

         - TODO: test with cell manager having 3 levels, plus root.

        """
        p_arrs = generate_sample_dataset_2()
        cm = CellManager(arrays_to_bin=p_arrs, min_cell_size=1.,
                         max_cell_size=2., num_levels=2)
        
        # the hierarchy would have been setup, we start issuing queries.
        cell_list = []
        # query for search particle 1.
        pnt = Point()
        pnt.x = 0.5
        pnt.y = 0.5
        pnt.z =  0.0

        cm.py_get_potential_cells(pnt, 0.1, cell_list)
        # cell_list should contain exactly four cells.
        self.assertEqual(len(cell_list), 4)
        id_list = []
        for cell in cell_list:
            self.assertEqual(cell.level, 0)
            id_list.append(cell.id)

        self.assertEqual(id_list.count(IntPoint(0, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(1, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(0, 1, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(-1, -1, 0)), 1)
        
        cell_list = []
        cm.py_get_potential_cells(pnt, 0.1, cell_list,
                                  single_layer=False)
                                  
        # exactly one cell should be returned.
        self.assertEqual(len(cell_list), 1)
        id_list = []
        for cell in cell_list:
            self.assertEqual(cell.level, 0)
            id_list.append(cell.id)
        self.assertEqual(id_list.count(IntPoint(0, 0, 0)), 1)

    def test_get_potential_cells_2(self):
        """
        Same as test_get_potential_cells_1.
        
        Tests for interaction radius of 0.5.
        """
        p_arrs = generate_sample_dataset_2()
        cm = CellManager(arrays_to_bin=p_arrs, min_cell_size=1.,
                         max_cell_size=2., num_levels=2)
        
        # the hierarchy would have been setup, we start issuing queries.
        cell_list = []
        # query for search particle 1.
        pnt = Point()
        pnt.x = 0.5
        pnt.y = 0.5
        pnt.z =  0.0

        cell_list = []
        cm.py_get_potential_cells(pnt, 0.5, cell_list)
        self.assertEqual(len(cell_list), 4)
        id_list = []
        for cell in cell_list:
            self.assertEqual(cell.level, 0)
            id_list.append(cell.id)

        self.assertEqual(id_list.count(IntPoint(0, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(1, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(0, 1, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(-1, -1, 0)), 1)

        cell_list = []
        cm.py_get_potential_cells(pnt, 0.5, cell_list,
                                  single_layer=False)

        # exactly one cell should be returned
        self.assertEqual(len(cell_list), 1)
        id_list = []
        for cell in cell_list:
            self.assertEqual(cell.level, 0)
            id_list.append(cell.id)

        self.assertEqual(id_list.count(IntPoint(0, 0, 0)), 1)

    def test_get_potential_cells_3(self):
        """
        Same as test_get_potential_cells_1
        
        Tests for a interaction radius of 1.0.
        """
        p_arrs = generate_sample_dataset_2()
        cm = CellManager(arrays_to_bin=p_arrs, min_cell_size=1.,
                         max_cell_size=2., num_levels=2)
        
        # the hierarchy would have been setup, we start issuing queries.
        cell_list = []
        # query for search particle 1.
        pnt = Point()
        pnt.x = 0.5
        pnt.y = 0.5
        pnt.z =  0.0

        cm.py_get_potential_cells(pnt, 1.0, cell_list)

        # four cells at level 0 (leaf) should be returned.
        self.assertEqual(len(cell_list), 4)
        id_list = []
        for cell in cell_list:
            self.assertEqual(cell.level, 0)
            id_list.append(cell.id)

        self.assertEqual(id_list.count(IntPoint(0, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(1, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(0, 1, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(-1, -1, 0)), 1)

        cell_list = []
        cm.py_get_potential_cells(pnt, 1.0, cell_list,
                                  single_layer=False)
        id_list = []
        # 4 cells should be returned.
        for cell in cell_list:
            self.assertEqual(cell.level, 0)
            id_list.append(cell.id)

        # check the cells returned.
        self.assertEqual(id_list.count(IntPoint(0, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(1, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(0, 1, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(-1, -1, 0)), 1)
        
    def test_get_potential_cells_4(self):
        """
        Same as test_get_potential_cells_1.

        Tests for a interaction radius of 2.0.
        """
        p_arrs = generate_sample_dataset_2()
        cm = CellManager(arrays_to_bin=p_arrs, min_cell_size=1.,
                         max_cell_size=2., num_levels=2)
        
        # the hierarchy would have been setup, we start issuing queries.
        cell_list = []
        # query for search particle 1.
        pnt = Point()
        pnt.x = 0.5
        pnt.y = 0.5
        pnt.z =  0.0

        cm.py_get_potential_cells(pnt, 2.0, cell_list)
        
        # this should return 5 cells from level 1.
        self.assertEqual(len(cell_list), 5)
        id_list = []

        for cell in cell_list:
            self.assertEqual(cell.level, 1)
            id_list.append(cell.id)

        # make sure the ids are proper.
        self.assertEqual(id_list.count(IntPoint(0, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(1, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(1,-1, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(-1, -1, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(-1, 1, 0)), 1)

        # now with single_layer off.
        cell_list = []
        id_list = []
        cm.py_get_potential_cells(pnt, 2.0, cell_list,
                                  single_layer=False)
        # 7 cells from level 0 should be returned.
        self.assertEqual(len(cell_list), 7)
        for cell in cell_list:
            self.assertEqual(cell.level, 0)
            id_list.append(cell.id)

        # make sure the ids are proper.
        self.assertEqual(id_list.count(IntPoint(0, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(1, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(2, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(-1, -1, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(2, -1, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(0, 1, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(-1, 2, 0)), 1)

    def test_get_potential_cells_5(self):
        """
        Same as test_get_potential_cells_1

        Tests for a interaction radius of 3.0.
        """
        p_arrs = generate_sample_dataset_2()
        cm = CellManager(arrays_to_bin=p_arrs, min_cell_size=1.,
                         max_cell_size=2., num_levels=2)
        
        # the hierarchy would have been setup, we start issuing queries.
        cell_list = []
        # query for search particle 1.
        pnt = Point()
        pnt.x = 0.5
        pnt.y = 0.5
        pnt.z =  0.0

        cm.py_get_potential_cells(pnt, 3.0, cell_list)

        # we should get exactly one cell, the root cell 
        self.assertEqual(len(cell_list), 1)
        self.assertEqual(cell_list[0], cm.root_cell)

        cell_list = []
        
        # now disable single layer search.
        cm.py_get_potential_cells(pnt, 3.0, cell_list,
                                  single_layer=False)
        # now cell_list should contain 5 cells at level 1
        self.assertEqual(len(cell_list), 5)
        id_list = []

        for cell in cell_list:
            self.assertEqual(cell.level, 1)
            id_list.append(cell.id)

        # make sure the ids are proper.
        self.assertEqual(id_list.count(IntPoint(0, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(1, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(1,-1, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(-1, -1, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(-1, 1, 0)), 1)
        
if __name__ == '__main__':
    unittest.main()
