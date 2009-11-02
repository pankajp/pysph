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
                               max_cell_size=2.0, num_levels=2)
    return dest, cell_manager


################################################################################
# `TestCellCache` class.
################################################################################
class TestCellCache(unittest.TestCase):
    """ 
    Tests the CellCache class.
    """
    def test_constructor(self):
        """
        """
        cell_cache = CellCache(None, None, 1.0, False, 'H')
        self.assertEqual(cell_cache.cell_manager, None)
        self.assertEqual(cell_cache.p_array, None)
        self.assertEqual(cell_cache.is_dirty, True)
        self.assertEqual(cell_cache.cache_list, [])
        self.assertEqual(cell_cache.radius_scale, 1.0)
        self.assertEqual(cell_cache.h, 'H')
        self.assertEqual(cell_cache.variable_h, False)

        pa, cm = generate_sample_dataset_2_nnps_test()
        # create a cache, with variable_h disabled and 
        # radius_scale = 1.0
        cell_cache = CellCache(cm, pa, 1.0, False)
        self.assertEqual(cell_cache.p_array, pa)
        self.assertEqual(cell_cache.cell_manager, cm)
        self.assertEqual(cell_cache.is_dirty, True)
        self.assertEqual(cell_cache.cache_list, [])
        self.assertEqual(cell_cache.radius_scale, 1.0)
        self.assertEqual(cell_cache.single_layer, True)
        self.assertEqual(cell_cache.variable_h, False)

        cell_cache = CellCache(cm, pa, 2.0, False)
        self.assertEqual(cell_cache.p_array, pa)
        self.assertEqual(cell_cache.cell_manager, cm)
        self.assertEqual(cell_cache.is_dirty, True)
        self.assertEqual(cell_cache.cache_list, [])
        self.assertEqual(cell_cache.radius_scale, 2.0)
        self.assertEqual(cell_cache.single_layer, False)
        self.assertEqual(cell_cache.variable_h, False)

        cell_cache = CellCache(cm, pa, 2.0, True)
        self.assertEqual(cell_cache.p_array, pa)
        self.assertEqual(cell_cache.cell_manager, cm)
        self.assertEqual(cell_cache.is_dirty, True)
        self.assertEqual(cell_cache.cache_list, [])
        self.assertEqual(cell_cache.radius_scale, 2.0)
        self.assertEqual(cell_cache.single_layer, False)
        self.assertEqual(cell_cache.variable_h, True)        

    def test_update(self):
        """
        Tests the update function.
        """
        pa, cm = generate_sample_dataset_2_nnps_test()
        cell_cache = CellCache(cm, pa, 1.0, False)

        x, y, z = pa.get('x', 'y', 'z')
        pnt = Point()
        
        cell_cache.py_update()

        self.assertEqual(cell_cache.is_dirty, False)
        self.assertEqual(len(cell_cache.cache_list), 7)
        
        # make sure each particle has its appropriate list of cells
        # present.
        cache_list = cell_cache.cache_list
        lst = []

        for i in range(7):
            pnt.x = x[i]
            pnt.y = y[i]
            pnt.z = z[i]

            lst[:] = []
            cm.py_get_potential_cells(pnt, 1.0, lst, True)
            self.assertEqual(len(cache_list[i]), len(lst))
            self.assertEqual(cache_list[i], lst)

        # now increase the radius of interaction, check the potential cells
        # returned.
        cell_cache = CellCache(cm, pa, 2.0, False)
        cell_cache.py_update()
        
        self.assertEqual(cell_cache.is_dirty, False)
        self.assertEqual(len(cell_cache.cache_list), 7)
        
        cache_list = cell_cache.cache_list
        
        for i in range(7):
            pnt.x = x[i]
            pnt.y = y[i]
            pnt.z = z[i]

            lst[:] = []
            cm.py_get_potential_cells(pnt, 2.0, lst, False)
            
            self.assertEqual(len(cache_list[i]), len(lst))
            self.assertEqual(cache_list[i], lst)

        cell_cache = CellCache(cm, pa, 5.0, False)
        cell_cache.py_update()

        self.assertEqual(cell_cache.is_dirty, False)
        self.assertEqual(len(cell_cache.cache_list), 7)
        
        cache_list = cell_cache.cache_list

        for i in range(7):
            pnt.x = x[i]
            pnt.y = y[i]
            pnt.z = z[i]
            
            lst[:] = []
            cm.py_get_potential_cells(pnt, 5.0, lst, False)
            
            self.assertEqual(len(cache_list[i]), len(lst))
            self.assertEqual(cache_list[i], lst)

    def test_get_potential_cells(self):
        """
        Tests the get_potential_cells function.
        """
        pa, cm = generate_sample_dataset_2_nnps_test()
        cell_cache = CellCache(cm, pa, 1.0, False)
        x, y, z = pa.get('x', 'y', 'z')
        pnt = Point()
        lst1 = []
        lst2 = []

        for i in range(7):
            pnt.x = x[i]
            pnt.y = y[i]
            pnt.z = z[i]
            
            lst1[:] = []
            lst2[:] = []
            
            cm.py_get_potential_cells(pnt, 1.0, lst1, True)
            cell_cache.py_get_potential_cells(i, lst2)

            self.assertEqual(len(lst1), len(lst2))
            self.assertEqual(lst1, lst2)

    def test_variable_h_cases(self):
        """
        Tests a variable-h input data.
        """
        msg = 'Variable-h test not implemented'
        raise NotImplementedError, msg


################################################################################
# `TestCellCacheManager` class.
################################################################################
class TestCellCacheManager(unittest.TestCase):
    """
    Tests the CellCacheManager class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        ccm = CellCacheManager()
        
        self.assertEqual(ccm.cell_manager, None)
        self.assertEqual(ccm.variable_h, False)
        self.assertEqual(ccm.h, 'h')

        self.assertEqual(ccm.is_dirty, True)
        self.assertEqual(len(ccm.cell_cache_dict), 0)

        ccm = CellCacheManager(variable_h=True, h='H')
        self.assertEqual(ccm.cell_manager, None)
        self.assertEqual(ccm.variable_h, True)
        self.assertEqual(ccm.h, 'H')

        self.assertEqual(ccm.is_dirty, True)
        self.assertEqual(len(ccm.cell_cache_dict), 0)

    def test_add_cache_entry(self):
        """
        Tests the add_cache_entry function.
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0, num_levels=1)

        ccm = CellCacheManager(cm)
        ccm.py_add_cache_entry(parrs[0], 1.0)
        
        self.assertEqual(len(ccm.cell_cache_dict), 1)
        ccm.py_add_cache_entry(parrs[0], 1.0)
        self.assertEqual(len(ccm.cell_cache_dict), 1)

        ccm.py_add_cache_entry(parrs[0], 2.0)
        self.assertEqual(len(ccm.cell_cache_dict), 2)

        ccm.py_add_cache_entry(parrs[1], 2.0)
        self.assertEqual(len(ccm.cell_cache_dict), 3)

    def test_update(self):
        """
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0, num_levels=1)

        ccm = CellCacheManager(cm)
        
        ccm.py_add_cache_entry(parrs[0], 1.0)
        ccm.py_add_cache_entry(parrs[0], 2.0)
        ccm.py_add_cache_entry(parrs[1], 2.0)
        ccm.py_add_cache_entry(parrs[1], 5.0)

        # although the parrays are themselves not dirty (reset by the
        # cell_manager) each of the individual caches will be dirty, as they
        # have not been updated yet.
        for cache in ccm.cell_cache_dict.values():
            cache.py_update()
            
        # call an update on ccm now
        ccm.py_update()

        # none of the caches should be dirty, as none of the parrays are dirty.
        for cache in ccm.cell_cache_dict.values():
            self.assertEqual(cache.is_dirty, False)

        # now set parrs[0] to dirty
        parrs[0].set_dirty(True)
        ccm.py_update()

        for cache in ccm.cell_cache_dict.values():
            self.assertEqual(cache.is_dirty, True)

        # manually update the caches.
        for cache in ccm.cell_cache_dict.values():
            cache.py_update()

        parrs[1].set_dirty(True)
        ccm.py_update()
        
        for cache in ccm.cell_cache_dict.values():
            self.assertEqual(cache.is_dirty, True)

        # manually update the caches.
        for cache in ccm.cell_cache_dict.values():
            cache.py_update()

        for cache in ccm.cell_cache_dict.values():
            self.assertEqual(cache.is_dirty, False)
        
    def test_get_cell_cache(self):
        """
        Tests the get_cell_cache function.
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0, num_levels=1)

        ccm = CellCacheManager(cm)
        ccm.py_add_cache_entry(parrs[0], 1.0)

        cc = ccm.py_get_cell_cache(parrs[0].name, 1.0)
        self.assertEqual(cc is not None, True)
        
        cc = ccm.py_get_cell_cache(parrs[1].name, 1.0)
        self.assertEqual(cc, None)
        
        cc = ccm.py_get_cell_cache(parrs[0].name, 1.0001)
        self.assertEqual(cc, None)

        cc = ccm.py_get_cell_cache(parrs[0].name, 1.0000000000001)
        self.assertEqual(cc, None)

    def test_variable_h_cases(self):
        """
        Test for variable-h input data.
        """
        msg = 'Variable-h test not implemented'
        raise NotImplementedError, msg

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
                         max_cell_size=2.0, num_levels=1)

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
                         max_cell_size=2.0, num_levels=1)

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
                         max_cell_size=2.0, num_levels=1)

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
                         max_cell_size=2.0, num_levels=1)

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
                         max_cell_size=2.0, num_levels=1)
        
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
                         max_cell_size=2.0, num_levels=1)

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
                         max_cell_size=2.0, num_levels=1)
        
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
                         max_cell_size=2.0, num_levels=1)
        
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
        nbrl = CachedNbrParticleLocator(None, None, 1.0, None, None)
        self.assertEqual(nbrl.radius_scale, 1.0)
        self.assertEqual(nbrl.source, None)
        self.assertEqual(nbrl.dest, None)
        self.assertEqual(nbrl.cell_cache, None)
        self.assertEqual(len(nbrl.particle_cache), 0)
        self.assertEqual(nbrl.h, 'h')

        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0, num_levels=1)
        
        cell_cache = CellCache(cm, parrs[1], 1.0, False, 'h')
        
        nbrl = CachedNbrParticleLocator(parrs[0], parrs[1], 1.0, cm, cell_cache)
        self.assertEqual(nbrl.radius_scale, 1.0)
        self.assertEqual(nbrl.source, parrs[0])
        self.assertEqual(nbrl.dest, parrs[1])
        self.assertEqual(nbrl.cell_cache, cell_cache)
        self.assertEqual(len(nbrl.particle_cache), 0)

    def test_update_status(self):
        """
        Tests the update_status function.
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0, num_levels=1)
        
        cell_cache = CellCache(cm, parrs[1], 1.0, False)
        nbrl = CachedNbrParticleLocator(parrs[0], parrs[1], 1.0, cm, cell_cache)
        
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
        nbrl = ConstHCachedNbrParticleLocator(None, None, 1.0, None, None)
        self.assertEqual(nbrl.radius_scale, 1.0)
        self.assertEqual(nbrl.source, None)
        self.assertEqual(nbrl.dest, None)
        self.assertEqual(nbrl.cell_cache, None)
        self.assertEqual(len(nbrl.particle_cache), 0)
        self.assertEqual(type(nbrl._locator), ConstHFixedDestNbrParticleLocator)
        
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0, num_levels=1)
        
        cell_cache = CellCache(cm, parrs[1], 1.0)
        
        nbrl = ConstHCachedNbrParticleLocator(parrs[0], parrs[1], 1.0, cm, cell_cache)
        self.assertEqual(nbrl.radius_scale, 1.0)
        self.assertEqual(nbrl.source, parrs[0])
        self.assertEqual(nbrl.dest, parrs[1])
        self.assertEqual(nbrl.cell_cache, cell_cache)
        self.assertEqual(len(nbrl.particle_cache), 0)

        cell_cache = CellCache(cm, parrs[1], 2.0)
        nbrl = ConstHCachedNbrParticleLocator(parrs[0], parrs[1], 2.0, cm, cell_cache)

    def test_update_status(self):
        """
        Tests the update_status function.
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0, num_levels=1)
        
        cell_cache = CellCache(cm, parrs[1], 1.0)
        nbrl = ConstHCachedNbrParticleLocator(parrs[0], parrs[1], 1.0, cm, cell_cache)
        
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
                         max_cell_size=2.0, num_levels=1)
        
        cell_cache = CellCache(cm, parrs[1], 1.0)
        
        nbrl = ConstHCachedNbrParticleLocator(parrs[0], parrs[1], 1.0, cm, cell_cache)
        nbrl.py_update()

        self.assertEqual(nbrl.is_dirty, False)
        # since caching has not been explicitly enabled, nothing will be cached.
        self.assertEqual(len(nbrl.particle_cache),
                         0)
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
                         max_cell_size=2.0, num_levels=1)
        
        cell_cache = CellCache(cm, parrs[1], 1.0)
        
        # test with no cell cache and no point caching.
        nbrl = ConstHCachedNbrParticleLocator(parrs[0], parrs[1], 2.0, cm,
                                        cell_cache=None, caching_enabled=False)
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
                                        cell_cache=cell_cache,
                                        caching_enabled=False)
        nbrl.py_update()
        nbrl.py_get_nearest_particles(0, a1, 2.0)
        self.assertEqual(set(a1.get_npy_array()), set(a2.get_npy_array()))

        # test with no cell cache, but point caching enabled.
        a1.reset()
        nbrl = ConstHCachedNbrParticleLocator(parrs[0], parrs[1], 2.0, cm,
                                        cell_cache=None,
                                        caching_enabled=True)
        nbrl.py_update()
        nbrl.py_get_nearest_particles(0, a1, 2.0)
        self.assertEqual(set(a1.get_npy_array()), set(a2.get_npy_array()))

        # test with both cell and point caching enabled.
        a1.reset()
        nbrl = ConstHCachedNbrParticleLocator(parrs[0], parrs[1], 2.0, cm,
                                        cell_cache=cell_cache,
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
        self.assertEqual(m.cell_cache_manager, None)
        self.assertEqual(m.use_cell_cache, False)
        self.assertEqual(m.variable_h, False)
        self.assertEqual(m.h, 'h')

        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0, num_levels=1)
        ccm = CellCacheManager(cm)

        m = CachedNbrParticleLocatorManager(cm, ccm, True, variable_h=False, h='h')
        self.assertEqual(m.cell_manager, cm)
        self.assertEqual(m.cell_cache_manager, ccm)
        self.assertEqual(m.use_cell_cache, True)
        self.assertEqual(m.variable_h, False)
        self.assertEqual(m.h, 'h')

    def test_add_interaction(self):
        """
        Test the add_interaction function.
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0, num_levels=1)
        ccm = CellCacheManager(cm)

        m = CachedNbrParticleLocatorManager(cm, ccm, True)
        
        m.add_interaction(parrs[0], parrs[1], 1.0)
        e = m.cache_dict.get((parrs[0].name, parrs[1].name, 1.0))
        self.assertEqual(e is not None, True)
        self.assertEqual(e.source, parrs[0])
        self.assertEqual(e.dest, parrs[1])
        self.assertEqual(e.radius_scale, 1.0)
        # because this interaction was added only once, caching should
        # not be enabled.
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
                         max_cell_size=2.0, num_levels=1)

        ccm = CellCacheManager(cm, variable_h=True)

        m = CachedNbrParticleLocatorManager(cm, ccm, True, variable_h=True)
        
        m.add_interaction(parrs[0], parrs[1], 1.0)
        e = m.cache_dict.get((parrs[0].name, parrs[1].name, 1.0))
        self.assertEqual(type(e), VarHCachedNbrParticleLocator)
        self.assertEqual(e.caching_enabled, False)
        m.add_interaction(parrs[0], parrs[1], 1.0)
        self.assertEqual(e.caching_enabled, True)

        m.add_interaction(parrs[0], parrs[1], 1.1)
        e1 = m.cache_dict.get((parrs[0].name, parrs[1].name, 1.1))
        self.assertEqual(e1 is not e, True)        
        
    def test_enable_disable_cell_cache_usage(self):
        """
        Tests the enable/disable_cell_cache_usage functions.
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0, num_levels=1)
        ccm = CellCacheManager(cm)
        
        m = CachedNbrParticleLocatorManager(cm, ccm, True)

        self.assertEqual(m.use_cell_cache, True)
        
        # now any cache that is created should use the cell cache.
        m.add_interaction(parrs[0], parrs[1], 1.0)
        e = m.get_cached_locator(parrs[0].name, parrs[1].name, 1.0)
        self.assertEqual(e.cell_cache is not None, True)

        m.disable_cell_cache_usage()
        self.assertEqual(m.use_cell_cache, False)
        
        m.add_interaction(parrs[0], parrs[1], 2.0)
        e = m.get_cached_locator(parrs[0].name, parrs[1].name, 2.0)
        self.assertEqual(e.cell_cache, None)
        # the previously added interaction should also have their
        # caches disabled.
        e = m.get_cached_locator(parrs[0].name, parrs[1].name, 1.0)
        self.assertEqual(e.cell_cache, None)

        m.enable_cell_cache_usage()
        self.assertEqual(m.use_cell_cache, True)
        m.add_interaction(parrs[0], parrs[0], 1.0)
        e = m.get_cached_locator(parrs[0].name, parrs[0].name, 1.0)
        self.assertEqual(e.cell_cache is not None, True)
        # add previous caches, should have cell caching enabled.
        e1 = m.get_cached_locator(parrs[0].name, parrs[1].name, 1.0)
        self.assertEqual(e1.cell_cache is not None, True)
        e2 = m.get_cached_locator(parrs[0].name, parrs[1].name, 2.0)
        self.assertEqual(e2.cell_cache is not None, True)
        
    def test_update(self):
        """
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0, num_levels=1)
        ccm = CellCacheManager(cm)
        
        m = CachedNbrParticleLocatorManager(cm, ccm, True)
        m.add_interaction(parrs[0], parrs[1], 1.0)
        m.add_interaction(parrs[0], parrs[1], 1.1)
        m.add_interaction(parrs[1], parrs[0], 1.0)

        # force update all the cahces
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
                         max_cell_size=2.0, num_levels=1)
        ccm = CellCacheManager(cm)
        
        m = CachedNbrParticleLocatorManager(cm, ccm, True)
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

        self.assertEqual(nm.cell_caching, True)
        self.assertEqual(nm.particle_caching, True)
        self.assertEqual(nm.polygon_caching, True)
        self.assertEqual(nm.cell_manager, None)
        self.assertEqual(nm.variable_h, False)
        self.assertEqual(nm.h, 'h')

        self.assertEqual(nm.cell_cache_manager is not None, True)
        self.assertEqual(nm.particle_cache_manager is not None, True)
        self.assertEqual(nm.polygon_cache_manager is not None, True)

        nm = NNPSManager(cell_caching=False, particle_caching=False,
                         polygon_caching=False, variable_h=True, h='H')
        self.assertEqual(nm.cell_caching, False)
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
                         max_cell_size=2.0, num_levels=1)

        # create an nnps manager with all types of caching disabled.
        nm = NNPSManager(cell_manager=cm, cell_caching=False,
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
        self.assertEqual(nl2.cell_cache, None)

        # enable cell caching also
        nm.enable_cell_caching()
        
        # make sure the particle cache manager used within has its cell_caching
        # set
        self.assertEqual(nm.particle_cache_manager.use_cell_cache, True)

        # all cached locators added previously should also have cell caching
        # this however is taken care by the CachedNbrParticleLocatorManager
        
        nl = nm.get_neighbor_particle_locator(parrs[0])
        self.assertEqual(type(nl), NbrParticleLocatorBase)
        
        nl = nm.get_neighbor_particle_locator(parrs[0], parrs[1])
        self.assertEqual(type(nl), ConstHFixedDestNbrParticleLocator)

        nl1 = nm.get_neighbor_particle_locator(parrs[0], parrs[1], 1.0)
        self.assertEqual(type(nl1), ConstHCachedNbrParticleLocator)
        self.assertEqual(nl1.cell_cache is not None, True)
        
        # now disable cell caching
        nm.disable_cell_caching()
        self.assertEqual(nm.particle_cache_manager.use_cell_cache, False)

        # now test for the variable h case ====================================
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0, num_levels=1)

        # create an nnps manager with all types of caching disabled.
        nm = NNPSManager(cell_manager=cm, cell_caching=False,
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
        self.assertEqual(nl2.cell_cache, None)

        # enable cell caching also
        nm.enable_cell_caching()
        
        # make sure the particle cache manager used within has its cell_caching
        # set
        self.assertEqual(nm.particle_cache_manager.use_cell_cache, True)

        # all cached locators added previously should also have cell caching
        # this however is taken care by the CachedNbrParticleLocatorManager
        
        nl = nm.get_neighbor_particle_locator(parrs[0])
        self.assertEqual(type(nl), NbrParticleLocatorBase)
        
        nl = nm.get_neighbor_particle_locator(parrs[0], parrs[1])
        self.assertEqual(type(nl), VarHFixedDestNbrParticleLocator)

        nl1 = nm.get_neighbor_particle_locator(parrs[0], parrs[1], 1.0)
        self.assertEqual(type(nl1), VarHCachedNbrParticleLocator)
        self.assertEqual(nl1.cell_cache is not None, True)
        
        # now disable cell caching
        nm.disable_cell_caching()
        self.assertEqual(nm.particle_cache_manager.use_cell_cache, False)

    def test_update(self):
        """
        Tests the update function.
        """
        parrs = generate_sample_dataset_1()
        cm = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                         max_cell_size=2.0, num_levels=1)

        # create an nnps manager with all types of caching disabled.
        nm = NNPSManager(cell_manager=cm, cell_caching=True,
                         particle_caching=True, polygon_caching=False)

        nm.get_neighbor_particle_locator(parrs[0], parrs[0], 1.0)
        nm.get_neighbor_particle_locator(parrs[0], parrs[1], 2.0)

        nm.py_update()

        # force update all the caches
        for c in nm.particle_cache_manager.cache_dict.values():
            c.py_update()
        for c in nm.cell_cache_manager.cell_cache_dict.values():
            c.py_update()

        # now mark parrs[0] as dirty.
        parrs[0].set_dirty(True)

        nm.py_update()

        # make sure that the caches have been marked dirty.
        for c in nm.particle_cache_manager.cache_dict.values():
            self.assertEqual(c.is_dirty, True)

        for c in nm.cell_cache_manager.cell_cache_dict.values():
            self.assertEqual(c.is_dirty, True)        

if __name__ == '__main__':
    unittest.main()


