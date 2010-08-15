"""
Tests for the nnps module.
"""
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
    Generate data like generate_sample_dataset_2, but with some extra
    information for testing the nnps module.
    """
    dest = generate_sample_dataset_2()[0]
    cell_manager = CellManager(arrays_to_bin=[dest], min_cell_size=1.,
                               max_cell_size=2.0)
    return dest, cell_manager


class TestNbrParticleLocatorBase(unittest.TestCase):
    """
    Tests the NbrParticleLocatorBase class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
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
        """
        Tests the get_nearest_particles_to_point function.

        For a graphical view of the test dataset, refer image
        test_cell_data1.png.

        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)

        nbrl1 = NbrParticleLocatorBase(parrs[0], cm)

        pnt = Point()
        # querying neighbors of dark point 4.(refer image)
        pnt.x = 0.4
        pnt.y = 0.0
        pnt.z = 0.4

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
        pnt.x = 1.5
        pnt.y = 0.0
        pnt.z = -0.5
        output_array.reset()
        nbrl1.py_get_nearest_particles_to_point(pnt, 4.0, output_array)
        self.assertEqual(output_array.length, 8)
        a = list(output_array.get_npy_array())
        for i in range(8):
            self.assertEqual(a.count(i), 1)

        # now querying for neighbors from dark particles.
        nbrl2 = NbrParticleLocatorBase(parrs[1], cm)
        
        # searching from the center (1., 1., 1.) with different radii.
        pnt.x = 1.
        pnt.y = 1.
        pnt.z = 1.
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


################################################################################
# `TestFixedDestinationNbrParticleLocator` class.
################################################################################
class TestFixedDestinationNbrParticleLocator(unittest.TestCase):
    """
    Tests the FixedDestinationNbrParticleLocator class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)

        nbrl = FixedDestinationNbrParticleLocator(parrs[0], parrs[1], cm, 'h') 
        
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
        """
        Tests the get_nearest_particles.
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)

        nbrl = FixedDestinationNbrParticleLocator(parrs[0], parrs[1], cm, 'h') 
        
        self.assertRaises(NotImplementedError, nbrl.py_get_nearest_particles,
                          0, None, 1.0, False)

################################################################################
# `TestConstHFixedDestNbrParticleLocator` class.
################################################################################
class TestConstHFixedDestNbrParticleLocator(unittest.TestCase):
    """
    Tests the ConstHFixedDestNbrParticleLocator class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)
        
        nbrl = ConstHFixedDestNbrParticleLocator(parrs[0], parrs[1], cm, 'h')  
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
        """
        Tests the get_nearest_particles function.

        The tests are essentially the same as the first set of tests
        of the test_get_nearest_particles_to_point function.
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)

        nbrl = ConstHFixedDestNbrParticleLocator(parrs[0], parrs[1], cm, 'h')

        output_array = LongArray()

        # querying neighbors of dark point 4, with radius 4.0
        nbrl.py_get_nearest_particles(3, output_array, 0.5)
        self.assertEqual(output_array.length, 1)
        self.assertEqual(output_array[0], 0)
        output_array.reset()

        nbrl.py_get_nearest_particles(3, output_array, 1.0)
        self.assertEqual(output_array.length, 4)
        a = list(output_array.get_npy_array())

        for i in range(4):
            self.assertEqual(a.count(i), 1)

        # querying neighbors of dark point 3, with radius 4.0
        output_array.reset()
        nbrl.py_get_nearest_particles(2, output_array, 4.0)
        self.assertEqual(output_array.length, 8)
        a = list(output_array.get_npy_array())
        for i in range(8):
            self.assertEqual(a.count(i), 1)

################################################################################
# `TestVarHFixedDestNbrParticleLocator` class.
################################################################################
class TestVarHFixedDestNbrParticleLocator(unittest.TestCase):
    """
    Tests the VarHFixedDestNbrParticleLocator.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)
        
        nbrl = VarHFixedDestNbrParticleLocator(parrs[0], parrs[1], cm, 'h')  
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
        """
        Tests the get_nearest_particles function.
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)
        
        nbrl = VarHFixedDestNbrParticleLocator(parrs[0], parrs[1], cm, 'h')  

        self.assertRaises(NotImplementedError, nbrl.py_get_nearest_particles,
                          0, None, 1.0, False)
        
        msg = 'VarHFixedDestNbrParticleLocator::get_nearest_particles'
        raise NotImplementedError, msg


################################################################################
# `TestCachedNbrParticleLocator` class.
################################################################################
class TestCachedNbrParticleLocator(unittest.TestCase):
    """
    Tests the CachedNbrParticleLocator class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        nbrl = CachedNbrParticleLocator(None, None, 1.0, None)
        self.assertEqual(nbrl.radius_scale, 1.0)
        self.assertEqual(nbrl.source, None)
        self.assertEqual(nbrl.dest, None)
        self.assertEqual(len(nbrl.particle_cache), 0)
        self.assertEqual(nbrl.h, 'h')

        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)
        
        nbrl = CachedNbrParticleLocator(parrs[0], parrs[1], 1.0, cm)
        self.assertEqual(nbrl.radius_scale, 1.0)
        self.assertEqual(nbrl.source, parrs[0])
        self.assertEqual(nbrl.dest, parrs[1])
        self.assertEqual(len(nbrl.particle_cache), 0)

    def test_update_status(self):
        """
        Tests the update_status function.
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)
        
        nbrl = CachedNbrParticleLocator(parrs[0], parrs[1], 1.0, cm)
        
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

class TestConstHCachedNbrParticleLocator(unittest.TestCase):
    """
    Tests the CachedNbrParticleLocator class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        nbrl = ConstHCachedNbrParticleLocator(None, None, 1.0, None)
        self.assertEqual(nbrl.radius_scale, 1.0)
        self.assertEqual(nbrl.source, None)
        self.assertEqual(nbrl.dest, None)
        self.assertEqual(len(nbrl.particle_cache), 0)
        self.assertEqual(type(nbrl._locator), ConstHFixedDestNbrParticleLocator)
        
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)
        
        nbrl = ConstHCachedNbrParticleLocator(parrs[0], parrs[1], 1.0, cm)
        self.assertEqual(nbrl.radius_scale, 1.0)
        self.assertEqual(nbrl.source, parrs[0])
        self.assertEqual(nbrl.dest, parrs[1])
        self.assertEqual(len(nbrl.particle_cache), 0)

        nbrl = ConstHCachedNbrParticleLocator(parrs[0], parrs[1], 2.0, cm)

    def test_update_status(self):
        """
        Tests the update_status function.
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)
        
        nbrl = ConstHCachedNbrParticleLocator(parrs[0], parrs[1], 1.0, cm)
        
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
        """
        Tests the update function.
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)
        
        nbrl = ConstHCachedNbrParticleLocator(parrs[0], parrs[1], 1.0, cm)
        nbrl.py_update()

        self.assertEqual(nbrl.is_dirty, False)
        # since caching has not been explicitly enabled, nothing will be cached.
        self.assertEqual(len(nbrl.particle_cache), 0)
        nbrl.enable_caching()
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
        self.assertEqual(len(nbrl.particle_cache),
                         2)

    def test_get_nearest_particles(self):
        """
        Tests the get_nearest_particles function.
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)
        
        # test with no cell cache and no point caching.
        nbrl = ConstHCachedNbrParticleLocator(parrs[0], parrs[1], 2.0, cm,
                                        caching_enabled=False)
        nbrl.py_update()

        fdnpl = ConstHFixedDestNbrParticleLocator(parrs[0], parrs[1], cm)
        a1 = LongArray()
        a2 = LongArray()

        nbrl.py_get_nearest_particles(0, a1, 2.0)
        fdnpl.py_get_nearest_particles(0, a2, 2.0)

        self.assertEqual(set(a1.get_npy_array()), set(a2.get_npy_array()))

        # test with cell cache, but no point caching.
        a1.reset()
        nbrl = ConstHCachedNbrParticleLocator(parrs[0], parrs[1], 2.0, cm,
                                        caching_enabled=False)
        nbrl.py_update()
        nbrl.py_get_nearest_particles(0, a1, 2.0)
        self.assertEqual(set(a1.get_npy_array()), set(a2.get_npy_array()))

        # test with no cell cache, but point caching enabled.
        a1.reset()
        nbrl = ConstHCachedNbrParticleLocator(parrs[0], parrs[1], 2.0, cm,
                                        caching_enabled=True)
        nbrl.py_update()
        nbrl.py_get_nearest_particles(0, a1, 2.0)
        self.assertEqual(set(a1.get_npy_array()), set(a2.get_npy_array()))

        # test with both cell and point caching enabled.
        a1.reset()
        nbrl = ConstHCachedNbrParticleLocator(parrs[0], parrs[1], 2.0, cm,
                                        caching_enabled=True)
        nbrl.py_update()
        nbrl.py_get_nearest_particles(0, a1, 2.0)
        self.assertEqual(set(a1.get_npy_array()), set(a2.get_npy_array()))

        # test for another point with both enabled.
        a1.reset()
        a2.reset()
        nbrl.py_get_nearest_particles(3, a1, 2.0)
        fdnpl.py_get_nearest_particles(3, a2, 2.0)
        self.assertEqual(set(a1.get_npy_array()), set(a2.get_npy_array()))


################################################################################
# `TestCachedNbrParticleLocatorManager` class.
################################################################################
class TestCachedNbrParticleLocatorManager(unittest.TestCase):
    """
    Tests the CachedNbrParticleLocatorManager class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        m = CachedNbrParticleLocatorManager()
        self.assertEqual(m.cell_manager, None)
        self.assertEqual(m.variable_h, False)
        self.assertEqual(m.h, 'h')

        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)

        m = CachedNbrParticleLocatorManager(cm, variable_h=False, h='h')
        self.assertEqual(m.cell_manager, cm)
        self.assertEqual(m.variable_h, False)
        self.assertEqual(m.h, 'h')

    def test_add_interaction(self):
        """
        Test the add_interaction function.
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)

        m = CachedNbrParticleLocatorManager(cm)
        print m.cache_dict
        m.add_interaction(parrs[0], parrs[1], 1.0)
        print m.cache_dict
        e = m.cache_dict.get((parrs[0].name, parrs[1].name, 1.0))
        print e
        self.assertEqual(e is not None, True)
        self.assertEqual(e.source, parrs[0])
        self.assertEqual(e.dest, parrs[1])
        self.assertEqual(e.radius_scale, 1.0)
        # because this interaction was added only once, caching should
        # not be enabled.
        print 'dirty', e.is_dirty
        self.assertEqual(e.caching_enabled, False)
        m.add_interaction(parrs[0], parrs[1], 1.0)
        self.assertEqual(e.caching_enabled, True)

        m.add_interaction(parrs[0], parrs[1], 1.1)
        e1 = m.cache_dict.get((parrs[0].name, parrs[1].name, 1.1))
        self.assertEqual(e1.radius_scale, 1.1)
        
        self.assertEqual(e1 is not e, True)

        # now test for variable_h case.
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)

        m = CachedNbrParticleLocatorManager(cm, variable_h=True)
        
        m.add_interaction(parrs[0], parrs[1], 1.0)
        e = m.cache_dict.get((parrs[0].name, parrs[1].name, 1.0))
        self.assertEqual(type(e), VarHCachedNbrParticleLocator)
        self.assertEqual(e.caching_enabled, False)
        m.add_interaction(parrs[0], parrs[1], 1.0)
        self.assertEqual(e.caching_enabled, True)

        m.add_interaction(parrs[0], parrs[1], 1.1)
        e1 = m.cache_dict.get((parrs[0].name, parrs[1].name, 1.1))
        self.assertEqual(e1 is not e, True)        
        
    def test_update(self):
        """
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)
        
        m = CachedNbrParticleLocatorManager(cm)
        m.add_interaction(parrs[0], parrs[1], 1.0)
        m.add_interaction(parrs[0], parrs[1], 1.1)
        m.add_interaction(parrs[1], parrs[0], 1.0)

        # force update all the caches
        for c in m.cache_dict.values():
            c.py_update()

        # set parrs[0] dirty
        parrs[0].set_dirty(True)
        
        m.py_update()

        # make sure all the caches have been marked as dirty.
        for c in m.cache_dict.values():
            self.assertEqual(c.is_dirty, True)

    def test_get_cached_locator(self):
        """
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)
        
        m = CachedNbrParticleLocatorManager(cm)
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


################################################################################
# `TestNNPSManager` class.
################################################################################
class TestNNPSManager(unittest.TestCase):
    """
    Tests the NNPSManager class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        nm = NNPSManager()

        self.assertEqual(nm.particle_caching, True)
        self.assertEqual(nm.polygon_caching, True)
        self.assertEqual(nm.cell_manager, None)
        self.assertEqual(nm.variable_h, False)
        self.assertEqual(nm.h, 'h')

        self.assertEqual(nm.particle_cache_manager is not None, True)
        self.assertEqual(nm.polygon_cache_manager is not None, True)

        nm = NNPSManager(particle_caching=False,
                         polygon_caching=False, variable_h=True, h='H')
        self.assertEqual(nm.particle_caching, False)
        self.assertEqual(nm.polygon_caching, False)
        self.assertEqual(nm.variable_h, True)
        self.assertEqual(nm.h, 'H')
        
    def test_get_neighbor_particle_locator(self):
        """
        Tests the get_neighbor_particle_locator function, with
        all types of caching enabled/disabled. Thus, this effectively
        tests the enable/disable functions too.
        
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)

        # create an nnps manager with all types of caching disabled.
        nm = NNPSManager(cell_manager=cm,
                         particle_caching=False, polygon_caching=False)
        
        # now get a locator for a single source, without any destination
        nl = nm.get_neighbor_particle_locator(parrs[0])
        # nl should be a base class
        self.assertEqual(type(nl), NbrParticleLocatorBase)
        
        nl = nm.get_neighbor_particle_locator(parrs[0], parrs[1])
        self.assertEqual(type(nl), ConstHFixedDestNbrParticleLocator)

        nl1 = nm.get_neighbor_particle_locator(parrs[0], parrs[1], 1.0)
        self.assertEqual(type(nl1), ConstHFixedDestNbrParticleLocator)
        
        nl2 = nm.get_neighbor_particle_locator(parrs[0], parrs[1], 1.0)
        self.assertEqual(nl1 is not nl2, True)
        self.assertEqual(type(nl1), ConstHFixedDestNbrParticleLocator)
        
        # enable particle caching now
        nm.enable_particle_caching()
        
        nl = nm.get_neighbor_particle_locator(parrs[0])
        self.assertEqual(type(nl), NbrParticleLocatorBase)
        
        nl = nm.get_neighbor_particle_locator(parrs[0], parrs[1])
        self.assertEqual(type(nl), ConstHFixedDestNbrParticleLocator)
        
        nl1 = nm.get_neighbor_particle_locator(parrs[0], parrs[1], 1.0)
        self.assertEqual(type(nl1), ConstHCachedNbrParticleLocator)
        self.assertEqual(nl1.caching_enabled, False)
        nl2 = nm.get_neighbor_particle_locator(parrs[0], parrs[1], 1.0)
        self.assertEqual(nl1 is nl2, True)
        self.assertEqual(nl2.caching_enabled, True)

        # all cached locators added previously should also have cell caching
        # this however is taken care by the CachedNbrParticleLocatorManager
        
        nl = nm.get_neighbor_particle_locator(parrs[0])
        self.assertEqual(type(nl), NbrParticleLocatorBase)
        
        nl = nm.get_neighbor_particle_locator(parrs[0], parrs[1])
        self.assertEqual(type(nl), ConstHFixedDestNbrParticleLocator)

        nl1 = nm.get_neighbor_particle_locator(parrs[0], parrs[1], 1.0)
        self.assertEqual(type(nl1), ConstHCachedNbrParticleLocator)
        
        # now test for the variable h case ====================================
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)

        # create an nnps manager with all types of caching disabled.
        nm = NNPSManager(cell_manager=cm,
                         particle_caching=False, polygon_caching=False, 
                         variable_h=True, h='h')

        # now get a locator for a single source, without any destination
        nl = nm.get_neighbor_particle_locator(parrs[0])
        # nl should be a base class
        self.assertEqual(type(nl), NbrParticleLocatorBase)
        
        nl = nm.get_neighbor_particle_locator(parrs[0], parrs[1])
        self.assertEqual(type(nl), VarHFixedDestNbrParticleLocator)

        nl1 = nm.get_neighbor_particle_locator(parrs[0], parrs[1], 1.0)
        self.assertEqual(type(nl1), VarHFixedDestNbrParticleLocator)
        
        nl2 = nm.get_neighbor_particle_locator(parrs[0], parrs[1], 1.0)
        self.assertEqual(nl1 is not nl2, True)
        self.assertEqual(type(nl1), VarHFixedDestNbrParticleLocator)

        # enable particle caching now
        nm.enable_particle_caching()
        
        nl = nm.get_neighbor_particle_locator(parrs[0])
        self.assertEqual(type(nl), NbrParticleLocatorBase)
        
        nl = nm.get_neighbor_particle_locator(parrs[0], parrs[1])
        self.assertEqual(type(nl), VarHFixedDestNbrParticleLocator)
        
        nl1 = nm.get_neighbor_particle_locator(parrs[0], parrs[1], 1.0)
        self.assertEqual(type(nl1), VarHCachedNbrParticleLocator)
        self.assertEqual(nl1.caching_enabled, False)
        nl2 = nm.get_neighbor_particle_locator(parrs[0], parrs[1], 1.0)
        self.assertEqual(nl1 is nl2, True)
        self.assertEqual(nl2.caching_enabled, True)

        # all cached locators added previously should also have cell caching
        # this however is taken care by the CachedNbrParticleLocatorManager
        
        nl = nm.get_neighbor_particle_locator(parrs[0])
        self.assertEqual(type(nl), NbrParticleLocatorBase)
        
        nl = nm.get_neighbor_particle_locator(parrs[0], parrs[1])
        self.assertEqual(type(nl), VarHFixedDestNbrParticleLocator)

        nl1 = nm.get_neighbor_particle_locator(parrs[0], parrs[1], 1.0)
        self.assertEqual(type(nl1), VarHCachedNbrParticleLocator)
        
    def test_update(self):
        """
        Tests the update function.
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0)

        # create an nnps manager with all types of caching disabled.
        nm = NNPSManager(cell_manager=cm,
                         particle_caching=True, polygon_caching=False)

        nm.get_neighbor_particle_locator(parrs[0], parrs[0], 1.0)
        nm.get_neighbor_particle_locator(parrs[0], parrs[1], 2.0)

        nm.py_update()

        # force update all the caches
        for c in nm.particle_cache_manager.cache_dict.values():
            c.py_update()

        # now mark parrs[0] as dirty.
        parrs[0].set_dirty(True)

        nm.py_update()

        # make sure that the caches have been marked dirty.
        for c in nm.particle_cache_manager.cache_dict.values():
            self.assertEqual(c.is_dirty, True)

if __name__ == '__main__':
    unittest.main()


