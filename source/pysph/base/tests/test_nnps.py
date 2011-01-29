""" Tests for the nnps module. """
# standard imports.
import unittest
import numpy

# local imports
from pysph.base.nnps import *
from pysph.base.tests.common_data import *
from pysph.base.cell import *
from pysph.base.point import Point
from pysph.base.carray import *

def generate_sample_dataset_2_nnps_test():
    """
    Generate data like generate_sample_dataset_2 (in common_data.py),
    with some additional information for testing the nnps module.
    """
    dest = generate_sample_dataset_2()[0]
    cell_manager = CellManager(arrays_to_bin=[dest], min_cell_size=1.,
                               max_cell_size=2.0)
    return dest, cell_manager

class TestBruteForceNNPS(unittest.TestCase):
    """ Tests the brute-force nnps """
    def test_brute_force_nnps(self):
        """ Tests the brute-force nnps
        
        For a graphical view of the test dataset, refer image
        test_cell_data1.png.
        """
        nbr_indices = LongArray()
        nbr_distances = DoubleArray()
        
        parrs = generate_sample_dataset_1()
        xa = parrs[0].get('x')
        ya = parrs[0].get('y')
        za = parrs[0].get('z')
        
        pnt = Point(0.4, 0.0, 0.4)
        brute_force_nnps(pnt, 1.0, xa, ya, za, nbr_indices, nbr_distances)
        self.assertEqual(nbr_indices.length, 4)
        a = list(nbr_indices.get_npy_array())
        for i in range(4):
            self.assertEqual(a.count(i), 1)
        
        # now querying for neighbors from dark particles.
        xa = parrs[1].get('x')
        ya = parrs[1].get('y')
        za = parrs[1].get('z')
        
        # searching from the center (1., 1., 1.) with different radii.
        pnt = Point(1, 1, 1)
        
        nbr_indices.reset()
        nbr_distances.reset()
        brute_force_nnps(pnt, 1.4142135623730951, xa, ya, za,
                         nbr_indices, nbr_distances)
        self.assertEqual(nbr_indices.length, 3)
        a = list(nbr_indices.get_npy_array())
        self.assertEqual(a.count(1), 1)
        self.assertEqual(a.count(3), 1)
        self.assertEqual(a.count(0), 1)
        
        # test with exclude_index argument
        nbr_indices.reset()
        nbr_distances.reset()
        brute_force_nnps(pnt, 1.4142135623730951, xa, ya, za,
                         nbr_indices, nbr_distances, exclude_index=1)
        self.assertEqual(nbr_indices.length, 2)
        a = list(nbr_indices.get_npy_array())
        self.assertEqual(a.count(1), 0)
        self.assertEqual(a.count(3), 1)
        self.assertEqual(a.count(0), 1)
        
        
class TestNbrParticleLocatorBase(unittest.TestCase):
    """Tests the NbrParticleLocatorBase class. """
    def test_constructor(self):
        """Tests the constructor. """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)

        nbrl1 = NbrParticleLocatorBase(parrs[0], cm)
        self.assertEqual(nbrl1.source, parrs[0])
        self.assertEqual(nbrl1.cell_manager, cm)
        self.assertEqual(nbrl1.source_index, 0)

        nbrl2 = NbrParticleLocatorBase(parrs[1], cm)
        self.assertEqual(nbrl2.source, parrs[1])
        self.assertEqual(nbrl2.cell_manager, cm)
        self.assertEqual(nbrl2.source_index, 1)

    def test_get_nearest_particles_to_point(self):
        """Tests the get_nearest_particles_to_point function.

        For a graphical view of the test dataset, refer image
        test_cell_data1.png.

        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)

        nbrl1 = NbrParticleLocatorBase(parrs[0], cm)

        # querying neighbors of dark point 4.(refer image)
        pnt = Point(0.4, 0.0, 0.4)

        output_array = LongArray()
        nbrl1.py_get_nearest_particles_to_point(pnt, 0.5, output_array)
        self.assertEqual(output_array.length, 1)
        self.assertEqual(output_array[0], 0)
        output_array.reset()

        nbrl1.py_get_nearest_particles_to_point(pnt, 1.0, output_array)
        self.assertEqual(output_array.length, 4)
        a = list(output_array.get_npy_array())
        for i in range(4):
            self.assertEqual(a.count(i), 1)

        # querying neighbors of dark point 3, with radius 4.0
        pnt = Point(1.5, 0.0, -0.5)
        output_array.reset()
        nbrl1.py_get_nearest_particles_to_point(pnt, 4.0, output_array)
        self.assertEqual(output_array.length, 8)
        a = list(output_array.get_npy_array())
        for i in range(8):
            self.assertEqual(a.count(i), 1)

        # now querying for neighbors from dark particles.
        nbrl2 = NbrParticleLocatorBase(parrs[1], cm)
        
        # searching from the center (1., 1., 1.) with different radii.
        pnt.set(1, 1, 1)
        output_array.reset()
        nbrl2.py_get_nearest_particles_to_point(pnt, 0.1, output_array)
        self.assertEqual(output_array.length, 0)
        
        nbrl2.py_get_nearest_particles_to_point(pnt, 1.0, output_array)
        self.assertEqual(output_array.length, 0)
        
        nbrl2.py_get_nearest_particles_to_point(pnt, 1.136, output_array)
        self.assertEqual(output_array.length, 1)
        a = list(output_array.get_npy_array())
        self.assertEqual(a.count(1), 1)
        output_array.reset()

        nbrl2.py_get_nearest_particles_to_point(pnt, 1.358, output_array)
        self.assertEqual(output_array.length, 2)
        a = list(output_array.get_npy_array())
        self.assertEqual(a.count(1), 1)
        self.assertEqual(a.count(3), 1)
        output_array.reset()

        nbrl2.py_get_nearest_particles_to_point(pnt, 1.4142135623730951,
                                             output_array)
        self.assertEqual(output_array.length, 3)
        a = list(output_array.get_npy_array())
        self.assertEqual(a.count(1), 1)
        self.assertEqual(a.count(3), 1)
        self.assertEqual(a.count(0), 1)
        output_array.reset()

        nbrl2.py_get_nearest_particles_to_point(pnt, 1.88, output_array)
        self.assertEqual(output_array.length, 4)
        a = list(output_array.get_npy_array())
        for i in range(4):
            self.assertEqual(a.count(i), 1)

        # test to make sure the exclude_index parameter is considered.
        output_array.reset()
        nbrl2.py_get_nearest_particles_to_point(pnt, 1.88, output_array, 3)
        self.assertEqual(output_array.length, 3)
        a = list(output_array.get_npy_array())
        for i in range(3):
            self.assertEqual(a.count(i), 1)


##############################################################################
# `TestFixedDestNbrParticleLocator` class.
##############################################################################
class TestFixedDestNbrParticleLocator(unittest.TestCase):
    """Tests the FixedDestNbrParticleLocator class. """
    def test_constructor(self):
        """Tests the constructor. """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)

        nbrl = FixedDestNbrParticleLocator(parrs[0], parrs[1], 1.0, cm, 'h') 
        
        self.assertEqual(nbrl.source, parrs[0])
        self.assertEqual(nbrl.source_index, 0)
        self.assertEqual(nbrl.dest, parrs[1])
        self.assertEqual(nbrl.dest_index, 1)
        self.assertEqual(nbrl.cell_manager, cm)
        self.assertEqual(nbrl.h, 'h')
        self.assertEqual(nbrl.d_h, parrs[1].get_carray('h'))
        self.assertEqual(nbrl.d_x, parrs[1].get_carray(cm.coord_x))
        self.assertEqual(nbrl.d_y, parrs[1].get_carray(cm.coord_y))
        self.assertEqual(nbrl.d_z, parrs[1].get_carray(cm.coord_z))
    
        nbrl = FixedDestNbrParticleLocator(None, None, 1.0, None)
        self.assertEqual(nbrl.radius_scale, 1.0)
        self.assertEqual(nbrl.source, None)
        self.assertEqual(nbrl.dest, None)
        self.assertEqual(len(nbrl.particle_cache), 0)
        
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)
        
        nbrl = FixedDestNbrParticleLocator(parrs[0], parrs[1], 1.0, cm)
        self.assertEqual(nbrl.radius_scale, 1.0)
        self.assertEqual(nbrl.source, parrs[0])
        self.assertEqual(nbrl.dest, parrs[1])
        self.assertEqual(len(nbrl.particle_cache), 0)

    def test_get_nearest_particles(self):
        """Tests the get_nearest_particles. """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)

        nbrl = FixedDestNbrParticleLocator(parrs[0], parrs[1], 1.0, cm, 'h')
        nbrl4 = FixedDestNbrParticleLocator(parrs[0], parrs[1], 4.0, cm, 'h')
        nbrl05 = FixedDestNbrParticleLocator(parrs[0], parrs[1], 0.5, cm, 'h')
        
        # querying neighbors of dark point 4, with radius 0.5
        output_array = nbrl05.py_get_nearest_particles(3)
        self.assertEqual(output_array.length, 1)
        self.assertEqual(output_array[0], 0)
        
        # querying neighbors of dark point 4, with radius 1
        output_array = nbrl.py_get_nearest_particles(3)
        self.assertEqual(output_array.length, 4)
        a = list(output_array.get_npy_array())
        for i in range(4):
            self.assertEqual(a.count(i), 1)

        # querying neighbors of dark point 3, with radius 4.0
        output_array = nbrl4.py_get_nearest_particles(2)
        self.assertEqual(output_array.length, 8)
        a = list(output_array.get_npy_array())
        for i in range(8):
            self.assertEqual(a.count(i), 1)
        
        nbrl = FixedDestNbrParticleLocator(parrs[0], parrs[1], 2.0, cm)
        nbrl.py_update()
        
        bnpl = NbrParticleLocatorBase(parrs[0], cm)
        
        a2 = LongArray()
        xa, ya, za = parrs[1].get('x'), parrs[1].get('y'), parrs[1].get('z')
        
        pnt = Point(xa[0], ya[0], za[0])
        nbrl.py_update()
        a1 = nbrl.py_get_nearest_particles(0)
        bnpl.py_get_nearest_particles_to_point(pnt, 2.0, a2)
        
        self.assertEqual(set(a1.get_npy_array()), set(a2.get_npy_array()))
        
        # test for another point.
        pnt = Point(xa[3], ya[3], za[3])
        a1 = nbrl.py_get_nearest_particles(3)
        bnpl.py_get_nearest_particles_to_point(pnt, 2.0, a2)
        
        self.assertEqual(set(a1.get_npy_array()), set(a2.get_npy_array()))

    def test_update_status(self):
        """Tests the update_status function. """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)
        
        nbrl = FixedDestNbrParticleLocator(parrs[0], parrs[1], 1.0, cm)
        
        nbrl.is_dirty = False
        # set either of the particle arrays to dirty and make sure 
        # the locator is also set to dirty
        parrs[0].set_dirty(True)
        nbrl.py_update_status()
        self.assertEqual(nbrl.is_dirty, True)

        # it should continue to remain dirty, until an update is called.
        parrs[0].set_dirty(False)
        nbrl.py_update_status()
        self.assertEqual(nbrl.is_dirty, True)
        
        parrs[1].set_dirty(True)
        nbrl.is_dirty = False
        nbrl.py_update_status()
        self.assertEqual(nbrl.is_dirty, True)        
    
    def test_update(self):
        """Tests the update function. """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)
        
        nbrl = FixedDestNbrParticleLocator(parrs[0], parrs[1], 1.0, cm)
        nbrl.py_update()

        self.assertEqual(nbrl.is_dirty, False)
        
        nbrl.py_update()
        self.assertEqual(len(nbrl.particle_cache),
                         parrs[1].get_number_of_particles())        
        
        # remove two particle from parrs[1] and check if update is done.
        to_remove = LongArray(2)
        to_remove[0] = 0
        to_remove[1] = 1
        parrs[1].remove_particles(to_remove)

        nbrl.py_update_status()
        nbrl.py_update()
        self.assertEqual(len(nbrl.particle_cache), 2)


##############################################################################
# `TestVarHNbrParticleLocator` class.
##############################################################################
class TestVarHNbrParticleLocator(unittest.TestCase):
    """Tests the VarHNbrParticleLocator. """
    def test_constructor(self):
        """Tests the constructor. """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)
        
        nbrl = VarHNbrParticleLocator(parrs[0], parrs[1], 1.0, cm, 'h')  
        self.assertEqual(nbrl.source, parrs[0])
        self.assertEqual(nbrl.source_index, 0)
        self.assertEqual(nbrl.dest, parrs[1])
        self.assertEqual(nbrl.dest_index, 1)
        self.assertEqual(nbrl.cell_manager, cm)
        self.assertEqual(nbrl.h, 'h')
        self.assertEqual(nbrl.d_h, parrs[1].get_carray('h'))
        self.assertEqual(nbrl.d_x, parrs[1].get_carray(cm.coord_x))
        self.assertEqual(nbrl.d_y, parrs[1].get_carray(cm.coord_y))
        self.assertEqual(nbrl.d_z, parrs[1].get_carray(cm.coord_z))

    def test_get_nearest_particles(self):
        """Tests the get_nearest_particles function. """
        # tests for particle array with const h
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)
        
        nbrl = VarHNbrParticleLocator(parrs[0], parrs[1], 1.0, cm, 'h')
        nbrl4 = VarHNbrParticleLocator(parrs[0], parrs[1], 4.0, cm, 'h')
        nbrl05 = VarHNbrParticleLocator(parrs[0], parrs[1], 0.5, cm, 'h')
        
        # querying neighbors of dark point 4, with radius 0.5
        output_array = nbrl05.py_get_nearest_particles(3)
        self.assertEqual(output_array.length, 1)
        self.assertEqual(output_array[0], 0)

        # querying neighbors of dark point 4, with radius 1
        output_array = nbrl.py_get_nearest_particles(3)
        self.assertEqual(output_array.length, 4)
        a = list(output_array.get_npy_array())
        for i in range(4):
            self.assertEqual(a.count(i), 1)

        # querying neighbors of dark point 3, with radius 4.0
        output_array = nbrl4.py_get_nearest_particles(2)
        self.assertEqual(output_array.length, 8)
        a = list(output_array.get_npy_array())
        for i in range(8):
            self.assertEqual(a.count(i), 1)
        
        # tests for particle array with varying h
        
        # get the distance matrix of all the particles
        parrs = generate_sample_dataset_1()
        parrs[0].append_parray(parrs[1])
        pall = parrs[0]
        dist = get_distance_matrix_pa(pall)
        
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)
        
        # set the h of two arrays to be different
        parrs[0].get('h')[:] = 3.0
        parrs[0].set_dirty(True)
        
        nbrl = VarHNbrParticleLocator(parrs[0], parrs[1], 1.0, cm, 'h')
        nbrl4 = VarHNbrParticleLocator(parrs[0], parrs[1], 4.0, cm, 'h')
        nbrl05 = VarHNbrParticleLocator(parrs[0], parrs[1], 0.5, cm, 'h')
        
        # querying neighbors of dark point 3, with radius 1
        # since we have kept h1 very large (3), all particles in parr1 should
        # be neighbors
        output_array = nbrl.py_get_nearest_particles(2)
        self.assertEqual(output_array.length, 8)
        a = list(output_array.get_npy_array())
        for i in range(8):
            self.assertEqual(a.count(i), 1)
        
        # now check by setting h of parr2 to be 0.1
        parrs[1].get('h')[:] = 0.1
        parrs[1].set_dirty(True)
        
        # should again return all the points
        output_array = nbrl.py_get_nearest_particles(2)
        self.assertEqual(output_array.length, 8)
        a = list(output_array.get_npy_array())
        for i in range(8):
            self.assertEqual(a.count(i), 1)
        
        # now set the h of white particle 6 to 3 and all others to 0.1
        # all black particles should only have particle 6 as a neighbor
        parrs[1].get('h')[:] = 0.1
        parrs[1].set_dirty(True)
        parrs[0].get('h')[:] = 0.1
        parrs[0].get('h')[5] = 3
        parrs[0].set_dirty(True)
        nbrl.py_update_status()
        
        # should only return white point 6
        for j in range(4):
            output_array = nbrl.py_get_nearest_particles(j)
            self.assertEqual(output_array.length, 1)
            self.assertEqual(output_array[0], 5)
        

##############################################################################
# `TestNNPSManager` class.
##############################################################################
class TestNNPSManager(unittest.TestCase):
    """Tests the NNPSManager class. """
    def test_constructor(self):
        """Tests the constructor. """
        nm = NNPSManager()

        self.assertEqual(nm.cell_manager, None)
        self.assertEqual(nm.variable_h, False)
        self.assertEqual(nm.h, 'h')

        nm = NNPSManager(variable_h=True, h='H')
        self.assertEqual(nm.variable_h, True)
        self.assertEqual(nm.h, 'H')
        
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)

        nm = NNPSManager(cm, variable_h=False, h='h')
        self.assertEqual(nm.cell_manager, cm)
        self.assertEqual(nm.variable_h, False)
        self.assertEqual(nm.h, 'h')

    def test_get_neighbor_particle_locator(self):
        """
        Tests the get_neighbor_particle_locator function

        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)

        nm = NNPSManager(cell_manager=cm)
        
        # now get a locator for a single source, without any destination
        nl = nm.get_neighbor_particle_locator(parrs[0])
        # nl should be a base class
        self.assertEqual(type(nl), NbrParticleLocatorBase)
        
        nl = nm.get_neighbor_particle_locator(parrs[0], parrs[1])
        self.assertEqual(type(nl), FixedDestNbrParticleLocator)

        nl1 = nm.get_neighbor_particle_locator(parrs[0], parrs[1], 1.0)
        self.assertEqual(type(nl1), FixedDestNbrParticleLocator)
        
        nl2 = nm.get_neighbor_particle_locator(parrs[0], parrs[1], 2.0)
        self.assertEqual(nl1 is nl2, False)
        self.assertEqual(type(nl2), FixedDestNbrParticleLocator)
        
        # this should return cached locator
        nl2 = nm.get_neighbor_particle_locator(parrs[0], parrs[1], 1.0)
        self.assertEqual(nl1 is nl2, True)
        
        # now test for the variable h case
        
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)

        # create an nnps manager with variable-h.
        nm = NNPSManager(cell_manager=cm, variable_h=True, h='h')

        # now get a locator for a single source, without any destination
        nl = nm.get_neighbor_particle_locator(parrs[0])
        # nl should be a base class
        self.assertEqual(type(nl), NbrParticleLocatorBase)
        
        nl = nm.get_neighbor_particle_locator(parrs[0], parrs[1])
        self.assertEqual(type(nl), VarHNbrParticleLocator)

        nl1 = nm.get_neighbor_particle_locator(parrs[0], parrs[1], 1.0)
        self.assertEqual(type(nl1), VarHNbrParticleLocator)
        self.assertEqual(nl1 is nl, True)
        
        nl2 = nm.get_neighbor_particle_locator(parrs[0], parrs[1], 2.0)
        self.assertEqual(nl1 is not nl2, True)
        self.assertEqual(type(nl1), VarHNbrParticleLocator) 
        
        nl = nm.get_neighbor_particle_locator(parrs[0], parrs[1], 1.0)
        self.assertEqual(nl1 is nl, True)

    def test_update(self):
        """Tests the update function. """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)

        # create an nnps manager.
        nm = NNPSManager(cell_manager=cm)

        nm.get_neighbor_particle_locator(parrs[0], parrs[0], 1.0)
        nm.get_neighbor_particle_locator(parrs[0], parrs[1], 2.0)

        nm.py_update()

        # force update all the caches
        for c in nm.particle_locator_cache.values():
            c.py_update()

        # now mark parrs[0] as dirty.
        parrs[0].set_dirty(True)

        nm.py_update()

        # make sure that the caches have been marked dirty.
        for c in nm.particle_locator_cache.values():
            self.assertEqual(c.is_dirty, True)

    def test_add_interaction(self):
        """Test the add_interaction function. """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)
        
        m = NNPSManager(cm, variable_h=False)
        m.add_interaction(parrs[0], parrs[1], 1.0)

        e = m.particle_locator_cache.get((parrs[0].name, parrs[1].name, 1.0))
        self.assertEqual(e is not None, True)
        self.assertEqual(e.source, parrs[0])
        self.assertEqual(e.dest, parrs[1])
        self.assertEqual(e.radius_scale, 1.0)
        m.add_interaction(parrs[0], parrs[1], 1.0)

        m.add_interaction(parrs[0], parrs[1], 1.1)
        e1 = m.particle_locator_cache.get((parrs[0].name, parrs[1].name, 1.1))
        self.assertEqual(e1.radius_scale, 1.1)
        
        self.assertEqual(e1 is not e, True)

        # now test for variable_h case.
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)

        m = NNPSManager(cm, variable_h=True)
        
        m.add_interaction(parrs[0], parrs[1], 1.0)
        e = m.particle_locator_cache.get((parrs[0].name, parrs[1].name, 1.0))
        self.assertEqual(type(e), VarHNbrParticleLocator)
        
        m.add_interaction(parrs[0], parrs[1], 1.0)

        m.add_interaction(parrs[0], parrs[1], 1.1)
        e1 = m.particle_locator_cache.get((parrs[0].name, parrs[1].name, 1.1))
        self.assertEqual(e1 is not e, True)        

    def test_get_cached_locator(self):
        """Tests the get_cached_locator function. """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)
        
        m = NNPSManager(cm)
        m.add_interaction(parrs[0], parrs[1], 1.0)
        m.add_interaction(parrs[1], parrs[1], 2.0)
        m.add_interaction(parrs[1], parrs[0], 1.0)
        
        e = m.get_cached_locator(parrs[0].name, parrs[1].name, 1.0)
        self.assertEqual(e is not None, True)
        self.assertEqual(e.source, parrs[0])
        self.assertEqual(e.dest, parrs[1])
        self.assertEqual(e.radius_scale, 1.0)

        e = m.get_cached_locator(parrs[1].name, parrs[1].name, 2.0)
        self.assertEqual(e is not None, True)
        self.assertEqual(e.source, parrs[1])
        self.assertEqual(e.dest, parrs[1])
        self.assertEqual(e.radius_scale, 2.0)

        e = m.get_cached_locator(parrs[1].name, parrs[0].name, 1.0)
        self.assertEqual(e is not None, True)
        self.assertEqual(e.source, parrs[1])
        self.assertEqual(e.dest, parrs[0])
        self.assertEqual(e.radius_scale, 1.0)

        e = m.get_cached_locator(parrs[1].name, parrs[1].name, 4.0)
        self.assertEqual(e, None)

if __name__ == '__main__':
    unittest.main()
