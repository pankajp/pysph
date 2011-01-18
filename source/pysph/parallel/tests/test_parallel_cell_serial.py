""" Tests for the individual functions in ParallelCellManager """

import pysph.base.api as base
import pysph.parallel.api as parallel

import numpy
import unittest


class ParallelCellManagerTestCase(unittest.TestCase):
    """ Test case for the ParallelCellManagerTestCase """

    def setUp(self):
        """ The setup consists of a 2D square ([0,1] X [0,1]) with 25
        particles.


        """

        xc = numpy.arange(0,1.0, 0.2)
        x, y = numpy.meshgrid(xc,xc)

        self.x = x = x.ravel()
        self.y = y = y.ravel()
        self.h = h = numpy.ones_like(x) * 0.25

        dx = dy = 0.2
        self.dx = dx

        self.block_size = 0.5
        self.cell_size = 0.5

        self.pa = pa = base.get_particle_array(name="test", x=x, y=y, h=h)

        self.cm = cm = parallel.ParallelCellManager(arrays_to_bin=[pa,],
                                                    max_radius_scale=2.0,
                                                    dimension=2.0,
                                                    load_balancing=False,
                                                    initialize=False)

        self.block_000_indices = 0,1,2,5,6,7,10,11,12
        self.block_100_indices = 3,4,8,9,13,14
        self.block_010_indices = 15,16,17,20,21,22
        self.block_110_indices = 18,19,23,24
        
    def test_update_global_properties(self):
        """ Test the initialize function """

        cm = self.cm
        cm.update_global_properties()

        local_bounds_min = cm.local_bounds_min
        local_bounds_max = cm.local_bounds_max

        glb_bounds_min = cm.glb_bounds_min
        glb_bounds_max = cm.glb_bounds_max

        self.assertAlmostEqual(local_bounds_min[0], min(self.x), 10)
        self.assertAlmostEqual(local_bounds_min[1], min(self.y), 10)
        self.assertAlmostEqual(local_bounds_min[2], 0.0, 10)

        self.assertAlmostEqual(local_bounds_max[0], max(self.x), 10)
        self.assertAlmostEqual(local_bounds_max[1], max(self.y), 10)
        self.assertAlmostEqual(local_bounds_max[2], 0.0, 10)

        self.assertAlmostEqual(glb_bounds_min[0], min(self.x), 10)
        self.assertAlmostEqual(glb_bounds_min[1], min(self.y), 10)
        self.assertAlmostEqual(glb_bounds_min[2], 0.0, 10)

        self.assertAlmostEqual(glb_bounds_max[0], max(self.x), 10)
        self.assertAlmostEqual(glb_bounds_max[1], max(self.y), 10)
        self.assertAlmostEqual(glb_bounds_max[2], 0.0, 10)

    def test_compute_cell_size(self):
        """ Test The cell size creation """

        cm = self.cm
        cm.initialize()

        origin = cm.origin
        cell_size = cm.cell_size
        factor = cm.factor

        self.assertAlmostEqual(origin.x, 0.0, 10)
        self.assertAlmostEqual(origin.y, 0.0, 10)
        self.assertAlmostEqual(origin.z, 0.0, 10)

        self.assertAlmostEqual(cell_size, 0.5, 10)
        self.assertEqual(factor, 1)

    def test_build_cell(self):
        """ Test the building of the base cell """

        cm = self.cm

        cm.update_global_properties()

        cm.setup_origin()

        cm.compute_block_size(0)

        cm.compute_cell_size(0,0)

        cm.py_rebuild_array_indices()

        cm.py_setup_cells_dict()
        
        cm.setup_processor_map()

        cm._build_cell()

        cells_dict = cm.cells_dict

        ncells = len(cells_dict)
        self.assertEqual(ncells, 1)

        cell = cells_dict.values()[0]

        centroid = base.Point()
        cell.get_centroid(centroid)

        self.assertAlmostEqual(centroid.x, 0.25, 10)
        self.assertAlmostEqual(centroid.y, 0.25, 10)
        self.assertAlmostEqual(centroid.z, 0.25, 10)

        indices = []
        cell.get_particle_ids(indices)
        index_array = indices[0]

        index_array_numpy = index_array.get_npy_array()
        index_array_numpy.sort()

        np = len(index_array_numpy)
        self.assertEqual(np, 25)

        for i in range(np):
            self.assertEqual(index_array_numpy[i], i)


    def test_bin_particles(self):
        """ Test the cells update function """

        cm = self.cm
        pa = cm.arrays_to_bin[0]

        cm.update_global_properties()

        cm.setup_origin()

        cm.compute_block_size(0)

        cm.compute_cell_size(0,0)

        cm.py_rebuild_array_indices()

        cm.py_setup_cells_dict()
        
        cm.setup_processor_map()

        cm._build_cell()

        new_block_cells, remote_block_cells = cm.bin_particles()

        proc_map = cm.proc_map

        # At this point only block 000 is assigned to the block map

        # the remote block cells should thus be empty

        self.assertEqual(len(remote_block_cells), 0)

        index_lists = []

        # test for new block (1,0,0)

        bid = base.IntPoint(1,0,0)
        cell = new_block_cells.get(bid)[0]
        cell.get_particle_ids(index_lists)

        particles_in_cell = pa.extract_particles(index_lists[0])
        np = particles_in_cell.get_number_of_particles()

        for i in range(np):
            self.assertEqual(particles_in_cell.idx[i],
                             self.block_100_indices[i])

        # test for new block (0,1,0)

        index_lists = []
        bid = base.IntPoint(0,1,0)
        cell = new_block_cells.get(bid)[0]
        cell.get_particle_ids(index_lists)

        particles_in_cell = pa.extract_particles(index_lists[0])
        np = particles_in_cell.get_number_of_particles()

        for i in range(np):
            self.assertEqual(particles_in_cell.idx[i],
                             self.block_010_indices[i])

        # test for new block (1,1,0)

        index_lists = []
        bid = base.IntPoint(1,1,0)
        cell = new_block_cells.get(bid)[0]
        cell.get_particle_ids(index_lists)

        particles_in_cell = pa.extract_particles(index_lists[0])
        np = particles_in_cell.get_number_of_particles()

        for i in range(np):
            self.assertEqual(particles_in_cell.idx[i],
                             self.block_110_indices[i])

        # test for block (0,0,0)

        index_lists = []
        bid = base.IntPoint(0,0,0)
        cell = new_block_cells.get(bid)[0]
        cell.get_particle_ids(index_lists)

        particles_in_cell = pa.extract_particles(index_lists[0])
        idx = particles_in_cell.get('idx')
        idx.sort()
        
        np = particles_in_cell.get_number_of_particles()

        for i in range(np):
            self.assertEqual(idx[i],
                             self.block_000_indices[i])

        # new particles for neighbors should be an empty dict

        remote_particles = cm.create_new_particle_copies(remote_block_cells)
        self.assertEqual(len(remote_particles), 0)

if __name__ == "__main__":
    unittest.main()
