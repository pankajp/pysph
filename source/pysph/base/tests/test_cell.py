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
    """Test various functions in the module."""
    
    def test_real_to_int(self):
        """Tests the real_to_int function."""
        self.assertEqual(py_real_to_int(0.5, 1.0), 0)
        self.assertEqual(py_real_to_int(0.1, 1.0), 0)

        self.assertEqual(py_real_to_int(1.0, 1.0), 1)        
        self.assertEqual(py_real_to_int(1.5, 1.0), 1)
        self.assertEqual(py_real_to_int(1.9, 1.0), 1)
        
        self.assertEqual(py_real_to_int(2.1, 1.0), 2)
        self.assertEqual(py_real_to_int(2.6, 1.0), 2) 

    def test_find_cell_id(self):
        """Tests the find_cell_id function."""
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


class TestCell(unittest.TestCase):
    """Tests for the Cell base class."""
    
    def test_constructor(self):
        cell = Cell(IntPoint(0, 0, 0), cell_manager=None, cell_size=0.1)
        
        self.assertEqual(cell.id, IntPoint(0, 0, 0))
        self.assertEqual(cell.coord_x=='x', True)
        self.assertEqual(cell.coord_y=='y', True)
        self.assertEqual(cell.coord_z=='z', True)

        self.assertEqual(cell.cell_size == 0.1, True)
        self.assertEqual(cell.cell_manager == None, True)
        self.assertEqual(cell.origin == Point(0., 0, 0), True)
        self.assertEqual(cell.jump_tolerance, 1)
        self.assertEqual(cell.arrays_to_bin == [], True)

    def test_set_cell_manager(self):
        """Tests the set_cell_manager function."""
        cell_manager = CellManager([])

        cell = Cell(IntPoint(0, 0, 0), cell_manager=cell_manager, cell_size=0.1)

        self.assertEqual(cell.arrays_to_bin == [], True)
        self.assertEqual(cell.origin, cell_manager.origin, True)
        self.assertEqual(cell.coord_x, cell_manager.coord_x, True)
        self.assertEqual(cell.coord_y, cell_manager.coord_y, True)
        self.assertEqual(cell.coord_z, cell_manager.coord_z, True)
        
        
        cell = Cell(IntPoint(0, 0, 0), cell_manager=None, cell_size=0.1)

        p_arrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=p_arrs, initialize=False)
        cell.set_cell_manager(cm)

        self.assertEqual(len(cell.index_lists), 2)

    def test_get_centroid(self):
        """Tests the get_centroid function."""
        cell = Cell(IntPoint(0, 0, 0), cell_manager=None, cell_size=0.1)
        centroid = Point()
        cell.py_get_centroid(centroid)

        self.assertEqual(centroid.x, 0.05)
        self.assertEqual(centroid.y, 0.05)
        self.assertEqual(centroid.z, 0.05)

        cell = Cell(IntPoint(1, 2, 3), cell_manager=None, cell_size=0.5)
        cell.py_get_centroid(centroid)

        self.assertEqual(centroid.x, 0.75)
        self.assertEqual(centroid.y, 1.25)
        self.assertEqual(centroid.z, 1.75)
        
    def test_update(self):
        """
        Tests the update function.
        """
        p_arrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=p_arrs, initialize=False)
        
        cell = Cell(IntPoint(0, 0, 0), cell_manager=cm, cell_size=1.0, jump_tolerance=1)

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
        
        cell = Cell(IntPoint(0, 0, 0), cell_manager=cm, cell_size=1.0,
                        jump_tolerance=1)
        
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
        
        cell1 = Cell(IntPoint(0, 0, 0), cell_manager=cm, cell_size=1.0, jump_tolerance=1)
        cell2 = Cell(IntPoint(0, 0, 0), cell_manager=cm, cell_size=1.0, jump_tolerance=1)
        cell3 = Cell(IntPoint(0, 0, 0), cell_manager=cm, cell_size=1.0, jump_tolerance=1)
        
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
    

###############################################################################
# `TestCellManager` class.
###############################################################################
class TestCellManager(unittest.TestCase):
    """Tests for the CellManager class."""
    
    def generate_random_particle_data(self, num_arrays, num_particles):
        """Returns a list of particle arrays with random data."""
        ret = []
        name = ''
       
        for i in range(num_arrays):
            x = numpy.random.rand(num_particles)
            y = numpy.random.rand(num_particles)
            z = numpy.random.rand(num_particles)
            name = 'arr' + str(i)

            p_arr = ParticleArray(name, **{'x':{'data':x}, 'y':{'data':y}, 'z':{'data':z}})
            ret.append(p_arr)
        
        return ret

    def test_constructor(self):
        """Tests the constructor."""
        cm = CellManager(initialize=False)
        
        # Some checks that should hold prior to cell_manager initialization.
        self.assertEqual(cm.origin, Point(0, 0, 0))
        self.assertEqual(len(cm.array_indices), 0)
        self.assertEqual(len(cm.arrays_to_bin), 0)
        self.assertEqual(cm.min_cell_size, 0.1)
        self.assertEqual(cm.max_cell_size, 0.5)
        self.assertEqual(cm.jump_tolerance, 1)
        self.assertEqual(cm.coord_x, 'x')
        self.assertEqual(cm.coord_y, 'y')
        self.assertEqual(cm.coord_z, 'z')
        self.assertEqual(len(cm.cells_dict), 0)
        
        # now call initialize
        cm.py_initialize()
        
        # there should be no cells since there are no particles
        self.assertEqual(len(cm.cells_dict), 0)

    def test_rebuild_array_indices(self):
        """Tests the _rebuild_array_indices function."""
        p_arrs = self.generate_random_particle_data(3, 10)
        cm = CellManager(arrays_to_bin=p_arrs, initialize=False)

        cm.py_rebuild_array_indices()

        # make sure the array_indices member is set properly.
        self.assertEqual(len(cm.arrays_to_bin), 3)
        self.assertEqual(len(cm.array_indices), 3)

        self.assertEqual(cm.array_indices[p_arrs[0].name], 0)
        self.assertEqual(cm.array_indices[p_arrs[1].name], 1)
        self.assertEqual(cm.array_indices[p_arrs[2].name], 2)
        
    def test_compute_cell_size(self):
        """Tests the compute_cell_sizes function."""
        cm = CellManager(initialize=False)
        
        cm.py_compute_cell_size(cm.min_cell_size, cm.max_cell_size) 

        #self.assertEqual(cm.cell_sizes.length, 1)
        self.assertEqual(cm.cell_size, cm.min_cell_size)

        # trying out some more calls to compute_cell_sizes
        arr = DoubleArray(10)
        
        cell_size = cm.py_compute_cell_size(1, 10)

        self.assertTrue(cell_size, 1)

    def test_build_cell(self):
        """Tests the _build_cell function."""
        p_arrs = self.generate_random_particle_data(3, 10)
        cm = CellManager(arrays_to_bin=p_arrs, initialize=False)

        cm.py_rebuild_array_indices()
        cm.py_setup_cells_dict()
        cm.py_build_cell()

        self.assertEqual(len(cm.cells_dict), 1)

        # there should be the only cell
        cell = cm.cells_dict.values()[0]

        # the cell should have all the 30 particle indices
        self.assertEqual(cell.py_get_number_of_particles(), 30)

        # make sure the jump tolerance of the cell is set to max
        self.assertEqual(cell.jump_tolerance, INT_INF())

        p_arrs = self.generate_random_particle_data(3, 10)
        cm = CellManager(arrays_to_bin=p_arrs, initialize=False)

        cm.py_rebuild_array_indices()
        cm.py_setup_cells_dict()
        cm.py_build_cell()

        self.assertEqual(len(cm.cells_dict), 1)
        
    def test_data_set_1(self):
        """Test initialize with data from generate_sample_dataset_1."""
        p_arrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=p_arrs, min_cell_size=2.,
                         max_cell_size=2.)
        
        self.assertEqual(len(cm.cells_dict), 2)
        
        p_arrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=p_arrs, min_cell_size=1.,
                         max_cell_size=1.)
        
        self.assertEqual(len(cm.cells_dict), 7)

    def test_data_set_2(self):
        """Test initialize with data from generate_sample_dataset_2"""
        p_arrs = generate_sample_dataset_2()
        cm = CellManager(arrays_to_bin=p_arrs, min_cell_size=1.,
                         max_cell_size=2.)

        self.assertEqual(len(cm.cells_dict), 7)
        
        for cell in cm.cells_dict.values():
            self.assertEqual(cell.py_get_number_of_particles(), 1)

        # further checking is not needed, the update test of the RootCell
        # would have handled that.

    def test_update(self):
        """Tests the update function.

        Use the dataset generated by generate_sample_dataset_2.

        We move three particles,  num 3, num 5 and num 7.
        """
        p_arrs = generate_sample_dataset_2()
        cm = CellManager(arrays_to_bin=p_arrs, min_cell_size=2.,
                         max_cell_size=2.)
        
        # move particle 3 in +ve y direction by one unit.
        y = p_arrs[0].get_carray('y')
        curr_y_2 = y.get(2)
        y[2] = curr_y_2 + 1.0
        # move particle 7 in -ve x direction by one unit
        x = p_arrs[0].get_carray('x')
        curr_x_6 = x.get(6)
        x[6] = curr_x_6 - 1.0
        # move particle 4 in -ve y direction by one unit
        curr_y_4 = y.get(4)
        y[4] = curr_y_4 - 1.0

        # 5 cells are formed
        self.assertEqual(len(cm.cells_dict), 5)
        # cell 0,0,0 has 3 particles since we have not updated
        self.assertEqual(cm.cells_dict[IntPoint()].py_get_number_of_particles(), 3)
        
        # because we changed the data, ask cm to update_status
        cm.py_update_status()
        cm.py_update()

        # no change should occur, as we have NOT set the dirty bit of the
        # parray.
        self.assertEqual(len(cm.cells_dict), 5)

        self.assertEqual(cm.cells_dict.has_key(IntPoint(0, 0, 0)), True)
        self.assertEqual(cm.cells_dict.has_key(IntPoint(1, 0, 0)), True)
        self.assertEqual(cm.cells_dict.has_key(IntPoint(1, -1, 0)), True)
        self.assertEqual(cm.cells_dict.has_key(IntPoint(-1, -1, 0)), True)
        self.assertEqual(cm.cells_dict.has_key(IntPoint(-1, 1, 0)), True)

        cell = cm.cells_dict[IntPoint(0, 0, 0)]
        self.assertEqual(cell.py_get_number_of_particles(), 3)
        cell = cm.cells_dict[IntPoint(1, 0, 0)]
        self.assertEqual(cell.py_get_number_of_particles(), 1)
        cell = cm.cells_dict[IntPoint(1, -1, 0)]
        self.assertEqual(cell.py_get_number_of_particles(), 1)
        cell = cm.cells_dict[IntPoint(-1, -1, 0)]
        self.assertEqual(cell.py_get_number_of_particles(), 1)
        cell = cm.cells_dict[IntPoint(-1, 1, 0)]
        self.assertEqual(cell.py_get_number_of_particles(), 1)

        # now set it to dirty and check.
        p_arrs[0].set_dirty(True)

        cm.py_update_status()
        cm.py_update()

        self.assertEqual(len(cm.cells_dict), 6)

        # make sure the proper cells have been created.
        self.assertEqual(cm.cells_dict.has_key(IntPoint(0, 0, 0)), True)

        # cell 1,-1,0  should have been deleted.
        self.assertEqual(cm.cells_dict.has_key(IntPoint(1, -1, 0)), False)

        # two new cells should have been created.
        self.assertEqual(cm.cells_dict.has_key(IntPoint(0, -1, 0)), True)
        self.assertEqual(cm.cells_dict.has_key(IntPoint(0, 1, 0)), True)

        cell = cm.cells_dict[IntPoint(0, -1, 0)]
        self.assertEqual(cell.py_get_number_of_particles(), 2)

        cell = cm.cells_dict[IntPoint(0, 1, 0)]
        self.assertEqual(cell.py_get_number_of_particles(), 1)

    def test_get_potential_cells_1(self):
        """Tests the get_potential_cells function.
        
        Data used is from generate_sample_dataset_2.

        Search facilities that will be needed:
         
         - given point and interaction radius, find all cells, that are
           immediate neighbors to the cell containing the given point.
         - given point and interaction radius, find all cells, that are possibly
           within its interaction radius.
         - **All particles are assumed to have constant interaction radius**.

        """
        p_arrs = generate_sample_dataset_2()
        cm = CellManager(arrays_to_bin=p_arrs, min_cell_size=1.,
                         max_cell_size=1.)
        cm2 = CellManager(arrays_to_bin=p_arrs, min_cell_size=2.,
                         max_cell_size=2.)
        
        # the cells would have been setup, we start issuing queries.
        cell_list = []
        # query for search particle 1.
        pnt = Point()
        pnt.x = 0.5
        pnt.y = 0.5
        pnt.z =  0.0

        cm.py_get_potential_cells(pnt, 0.1, cell_list)
        # cell_list should contain exactly four cells.
        self.assertEqual(len(cell_list), 4)
        id_list = cm.cells_dict.keys()

        self.assertEqual(id_list.count(IntPoint(0, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(1, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(0, 1, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(-1, -1, 0)), 1)
        
    def test_get_potential_cells_2(self):
        """Same as test_get_potential_cells_1.
        
        Tests for interaction radius of 0.5.
        """
        p_arrs = generate_sample_dataset_2()
        cm = CellManager(arrays_to_bin=p_arrs, min_cell_size=1.,
                         max_cell_size=2.)
        
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
            id_list.append(cell.id)

        self.assertEqual(id_list.count(IntPoint(0, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(1, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(0, 1, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(-1, -1, 0)), 1)

    def test_get_potential_cells_3(self):
        """Same as test_get_potential_cells_1
        
        Tests for a interaction radius of 1.0.
        """
        p_arrs = generate_sample_dataset_2()
        cm = CellManager(arrays_to_bin=p_arrs, min_cell_size=1.,
                         max_cell_size=2.)
        
        # the cells would have been setup, we start issuing queries.
        cell_list = []
        # query for search particle 1.
        pnt = Point()
        pnt.x = 0.5
        pnt.y = 0.5
        pnt.z =  0.0

        cm.py_get_potential_cells(pnt, 1.0, cell_list)

        # 4 cells should be returned.
        self.assertEqual(len(cell_list), 4)
        id_list = []
        for cell in cell_list:
            id_list.append(cell.id)

        self.assertEqual(id_list.count(IntPoint(0, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(1, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(0, 1, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(-1, -1, 0)), 1)

        cell_list = []
        cm.py_get_potential_cells(pnt, 1.0, cell_list)
        id_list = []
        # 4 cells should be returned.
        for cell in cell_list:
            id_list.append(cell.id)

        # check the cells returned.
        self.assertEqual(id_list.count(IntPoint(0, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(1, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(0, 1, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(-1, -1, 0)), 1)
        
    def test_get_potential_cells_4(self):
        """Same as test_get_potential_cells_1.

        Tests for a interaction radius of 2.0.
        """
        p_arrs = generate_sample_dataset_2()
        cm = CellManager(arrays_to_bin=p_arrs, min_cell_size=2.,
                         max_cell_size=2.)
        
        # the hierarchy would have been setup, we start issuing queries.
        cell_list = []
        # query for search particle 1.
        pnt = Point()
        pnt.x = 0.5
        pnt.y = 0.5
        pnt.z =  0.0

        cm.py_get_potential_cells(pnt, 2.0, cell_list)
        
        # this should return 5 cells
        self.assertEqual(len(cell_list), 5)
        id_list = []

        for cell in cell_list:
            id_list.append(cell.id)

        # make sure the ids are proper.
        self.assertEqual(id_list.count(IntPoint(0, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(1, 0, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(1,-1, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(-1, -1, 0)), 1)
        self.assertEqual(id_list.count(IntPoint(-1, 1, 0)), 1)

    def test_get_potential_cells_5(self):
        """Same as test_get_potential_cells_1

        Tests for a interaction radius of 3.0.
        """
        p_arrs = generate_sample_dataset_2()
        cm = CellManager(arrays_to_bin=p_arrs, min_cell_size=2.,
                         max_cell_size=2.)
        
        # the cells would have been setup, we start issuing queries.
        cell_list = []
        # query for search particle 1.
        pnt = Point()
        pnt.x = 0.5
        pnt.y = 0.5
        pnt.z =  0.0

        cm.py_get_potential_cells(pnt, 3.0, cell_list)

        # we should get all the cells 
        self.assertEqual(len(cell_list), len(cm.cells_dict))
    
    def test_get_potential_cells_6(self):
        """Same as test_get_potential_cells_1
        
        Test for cell_size = 1.0 and interaction radius = 3.0.
        """
        p_arrs = generate_sample_dataset_2()
        cm = CellManager(arrays_to_bin=p_arrs, min_cell_size=1.,
                         max_cell_size=2.)
        
        # the cells would have been setup, we start issuing queries.
        cell_list = []
        # query for search particle 1.
        pnt = Point()
        pnt.x = 0.5
        pnt.y = 0.5
        pnt.z =  0.0

        cm.py_get_potential_cells(pnt, 3.0, cell_list)
        
        # we should get all the cells
        self.assertEqual(len(cell_list), len(cm.cells_dict))

    def test_cells_update(self):
        """Tests the update function."""
        p_arrs = generate_sample_dataset_2()
        cm = CellManager(arrays_to_bin=p_arrs, initialize=False)

        big_cell = Cell(IntPoint(0, 0, 0), cell_manager=cm,
                               cell_size=2.0)
                
        cm.cells_dict[big_cell.id.py_copy()] = big_cell

        # add all particle indices to big_cell
        indx_arr = big_cell.index_lists[0]
        indx_arr.resize(7)
        indx_arr.set_data(numpy.arange(7, dtype=numpy.int))

        cm.py_update()

        # now we should have a total of 5 cells.
        self.assertEqual(len(cm.cells_dict), 5)

        self.assertEqual(cm.cells_dict.has_key(IntPoint(-1, 1, 0)), True)
        self.assertEqual(cm.cells_dict.has_key(IntPoint(-1, -1, 0)), True)
        self.assertEqual(cm.cells_dict.has_key(IntPoint(1, 0, 0)), True)
        self.assertEqual(cm.cells_dict.has_key(IntPoint(1, -1, 0)), True)
        self.assertEqual(cm.cells_dict.has_key(IntPoint(0, 0, 0)), True)

        # we do not test further as those test would have been performed by
        # tests for NonLeafCell.

if __name__ == '__main__':
    unittest.main()
