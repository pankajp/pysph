"""
Tests for the sph_calc module.
"""
# standard imports
import unittest

# local imports
from pysph.sph.sph_calc import SPHBase
from pysph.base.kernel1d import *
from pysph.base.nnps import NNPSManager
from pysph.base.cell import CellManager
from pysph.sph.misc_particle_funcs import *
from pysph.sph.tests.common_data import *
from pysph.base.kernelbase import KernelBase
from pysph.base.carray import DoubleArray
from pysph.base.particle_tags import *


def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-16."""
    return numpy.allclose(x, y, atol=1e-16, rtol=0)


class TstKernel(KernelBase):
    """
    Dummy kernel used for some tests.
    """
    def radius(self):
        return 1.0

def get_sample_data(dimenion, output_fields):
    """
    Return appropriate test data.
    """
    parrs = None
    funcs = []
    if dimenion == 2:
        parrs = generate_sample_dataset_1()
    elif dimenion == 3:
        parrs = generate_sample_dataset_2()
    else:
        raise ValueError, 'dimenion should be 2 or 3'

    cell_man = CellManager(arrays_to_bin=parrs, min_cell_size=1.,
                               max_cell_size=2.0, num_levels=1)

    nnps_man = NNPSManager(cell_manager=cell_man)
    kernel = TstKernel()

    if output_fields == 1:
        funcs.append(NeighborCountFunc(parrs[0], parrs[0]))
    elif output_fields == 2:
        funcs.append(NeighborCountFunc2(parrs[0], parrs[0]))
    elif output_fields == 3:
        funcs.append(NeighborCountFunc3(parrs[0], parrs[0]))
    elif output_fields == 4:
        funcs.append(NeighborCountFunc4(parrs[0], parrs[0]))
    else:
        raise ValueError, 'output_fields should be 1/2/3/4'

    return parrs, funcs, nnps_man, kernel
    
class TestSPHBase(unittest.TestCase):
    """
    Tests the SPHBase class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        parrs, funcs, nnps_man, kernel = get_sample_data(2, 1)
        sph = SPHBase(parrs, parrs[0], kernel, funcs, nnps_man)

        self.assertEqual(sph.sources[0], parrs[0])
        self.assertEqual(sph.dest, parrs[0])
        self.assertEqual(sph.kernel, kernel)
        self.assertEqual(sph.nnps_manager, nnps_man)

        self.assertEqual(sph.sph_funcs[0], funcs[0])

        self.assertEqual(sph.h, 'h')
        self.assertEqual(sph.valid_call, 1)
        self.assertEqual(len(sph.nbr_locators), 1)

    def test_sph1_2d_dataset(self):
        """
        Test the sph1 and sph1_array function with the NeighborCountFunc with a
        2d dataset.
        """
        parrs, funcs, nnps_man, kernel = get_sample_data(2, 1)
        parrs[0].add_temporary_array('ncount')

        sph = SPHBase(parrs, parrs[0], kernel, funcs, nnps_man)

        # make sure the other calls disabled.
        self.assertRaises(ValueError, sph.sph2, None, None)
        self.assertRaises(ValueError, sph.sph3, None, None, None)
        self.assertRaises(ValueError, sph.sphn, [])

        arr = DoubleArray(parrs[0].get_number_of_particles())
        arr.get_npy_array()[:] = 0.0
        
        sph.sph1_array(arr, exclude_self=True)
        vals = [2., 3., 2., 3., 4., 3., 2., 3., 2.]
        self.assertEqual(check_array(vals, arr.get_npy_array()), True)
        
        arr.get_npy_array()[:] = 0
        sph.sph1_array(arr, exclude_self=False)
        vals = [3., 4., 3., 4., 5., 4., 3., 4., 3.]
        self.assertEqual(check_array(vals, arr.get_npy_array()), True)

        arr1 = parrs[0].get_carray('ncount')
        sph.sph1('ncount', exclude_self=False)
        
        vals = [3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 3.0, 4.0, 3.0]
        self.assertEqual(check_array(vals, arr1.get_npy_array()), True)

    def test_sph1_3d_dataset(self):
        """
        Test the sph1 and sph1_array function with the NeighborCountFunc with a
        3d dataset.
        """
        parrs, funcs, nnps_man, kernel = get_sample_data(3, 1)

        sph = SPHBase(parrs, parrs[0], kernel, funcs, nnps_man)
        
        arr = parrs[0].get_carray('t')
        arr.get_npy_array()[:] = 0.0

        sph.sph1('t', exclude_self=True)

        vals = [4., 4., 4., 4., 4., 4., 4., 4., 8.]
        self.assertEqual(check_array(arr.get_npy_array(), vals), True)

        arr.get_npy_array()[:] = 0.0

        sph.sph1('t', exclude_self=False)

        vals = [5., 5., 5., 5., 5., 5., 5., 5., 9.]
        self.assertEqual(check_array(vals, arr.get_npy_array()), True)
        
    def test_sph1_with_tags(self):
        """
        Test the effect of setting tags to particles.
        """
        parrs, funcs, nnps_man, kernel = get_sample_data(2, 1)
        parrs[0].add_temporary_array('ncount')

        sph = SPHBase(parrs, parrs[0], kernel, funcs, nnps_man)
        
        # change the tag of a few particles and make sure they are not computed.
        tags = parrs[0].get_carray('tag')
        tags[0] = get_remote_real_tag()
        tags[1] = get_remote_dummy_tag()
        tags[5] = get_remote_dummy_tag()
        tags[7] = get_local_dummy_tag()

        arr = parrs[0].get_carray('ncount')

        arr.get_npy_array()[:] = 0.0
        arr[0] = arr[1] = arr[5] = arr[7] = -1.0

        sph.sph1('ncount', exclude_self=False)

        vals=[-1., -1., 3., 4., 5., -1., 3., -1., 3.]
        self.assertEqual(check_array(vals, arr.get_npy_array()), True)

    def test_sph2_2d_dataset(self):
        """
        Test the sph2 and sph2_array function with the NeighborCountFunc2 with a
        2d dataset.
        """
        parrs, funcs, nnps_man, kernel = get_sample_data(2, 2)
        parrs[0].add_temporary_array('ncount1')
        parrs[0].add_temporary_array('ncount2')

        sph = SPHBase(parrs, parrs[0], kernel, funcs, nnps_man)
        
        arr1 = parrs[0].get_carray('ncount1')
        arr2 = parrs[0].get_carray('ncount2')

        arr1.get_npy_array()[:] = 0.0
        arr2.get_npy_array()[:] = 0.0

        # make sure the other sph* functions are not callable.
        self.assertRaises(ValueError, sph.sph1, None)
        self.assertRaises(ValueError, sph.sph3, None, None, None)
        self.assertRaises(ValueError, sph.sphn, [])
        
        sph.sph2('ncount1', 'ncount2', exclude_self=False)
        
        vals1= [3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 3.0, 4.0, 4.0]
        check_array(vals1, arr1.get_npy_array())
        check_array(vals1, arr2.get_npy_array())

        arr1.get_npy_array()[:] = 0.0
        arr2.get_npy_array()[:] = 0.0
        sph.sph2('ncount1', 'ncount2', exclude_self=True)
        
        vals1 = [2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 2.0]
        check_array(vals1, arr1.get_npy_array())
        check_array(vals1, arr2.get_npy_array())

    def test_sph2_3d_dataset(self):
        """
        Test the sph2 and sph2_array function with the NeighborCountFunc2 with a
        3d dataset.
        """
        parrs, funcs, nnps_man, kernel = get_sample_data(3, 2)
        parrs[0].add_temporary_array('ncount1')
        parrs[0].add_temporary_array('ncount2')
        
        arr1 = parrs[0].get_carray('ncount1')
        arr2 = parrs[0].get_carray('ncount2')

        arr1.get_npy_array()[:] = 0.0
        arr2.get_npy_array()[:] = 0.0

        sph = SPHBase(parrs, parrs[0], kernel, funcs, nnps_man)

        # make sure the other sph* functions are not callable.
        self.assertRaises(ValueError, sph.sph1, None)
        self.assertRaises(ValueError, sph.sph3, None, None, None)
        self.assertRaises(ValueError, sph.sphn, [])

        sph.sph2('ncount1', 'ncount2', exclude_self=True)

        vals = [4., 4., 4., 4., 4., 4., 4., 4., 8.]
        self.assertEqual(check_array(vals, arr1.get_npy_array()), True)
        self.assertEqual(check_array(vals, arr2.get_npy_array()), True)

        arr1.get_npy_array()[:] = 0.0
        arr2.get_npy_array()[:] = 0.0
        sph.sph2('ncount1', 'ncount2', exclude_self=False)
        
        vals = [5., 5., 5., 5., 5., 5., 5., 5., 9.]
        
        self.assertEqual(check_array(vals, arr1.get_npy_array()), True)
        self.assertEqual(check_array(vals, arr2.get_npy_array()), True)

    def test_sph3(self):
        """
        Test the sph3 and sph3_array function with the NeighborCountFunc3 func.
        Uses a 3d dataset.
        """
        parrs, funcs, nnps_man, kernel = get_sample_data(3, 3)
        parrs[0].add_temporary_array('ncount1')
        parrs[0].add_temporary_array('ncount2')
        parrs[0].add_temporary_array('ncount3')

        arr1 = parrs[0].get_carray('ncount1')
        arr2 = parrs[0].get_carray('ncount2')
        arr3 = parrs[0].get_carray('ncount3')

        arr1.get_npy_array()[:] = 0.0
        arr2.get_npy_array()[:] = 0.0
        arr3.get_npy_array()[:] = 0.0

        sph = SPHBase(parrs, parrs[0], kernel, funcs, nnps_man)

        self.assertRaises(ValueError, sph.sph1, None)
        self.assertRaises(ValueError, sph.sph2, None, None)
        self.assertRaises(ValueError, sph.sphn, [])
        
        sph.sph3('ncount1', 'ncount2', 'ncount3', exclude_self=True)

        vals = [4., 4., 4., 4., 4., 4., 4., 4., 8.]

        self.assertEqual(check_array(vals, arr1.get_npy_array()), True)
        self.assertEqual(check_array(vals, arr2.get_npy_array()), True)
        self.assertEqual(check_array(vals, arr3.get_npy_array()), True)
        
        arr1.get_npy_array()[:] = 0.0
        arr2.get_npy_array()[:] = 0.0
        arr3.get_npy_array()[:] = 0.0
        
        sph.sph3('ncount1', 'ncount2', 'ncount3', exclude_self=False)

        vals = [5., 5., 5., 5., 5., 5., 5., 5., 9.]
        self.assertEqual(check_array(vals, arr1.get_npy_array()), True)
        self.assertEqual(check_array(vals, arr2.get_npy_array()), True)
        self.assertEqual(check_array(vals, arr3.get_npy_array()), True)


    def test_sphn(self):
        """
        Test the sphn and sphn_array function with the NeighborCountFunc4
        function. Uses a 3d dataset.
        """
        parrs, funcs, nnps_man, kernel = get_sample_data(3, 4)
        
        arrs = []
        names= []
        # add 4 temporary arrys to hold the results, and get the arrays in a
        # list.
        for i in range(1,5):
            name = 'ncount' + str(i)
            names.append(name)
            parrs[0].add_temporary_array(name)
            arrs.append(parrs[0].get_carray(name))

        for i in range(4):
            arrs[i].get_npy_array()[:] = 0.0

        sph = SPHBase(parrs, parrs[0], kernel, funcs, nnps_man)
        # make sure the other calls don't work.
        self.assertRaises(ValueError, sph.sph1, None)
        self.assertRaises(ValueError, sph.sph2, None, None)
        self.assertRaises(ValueError, sph.sph3, None, None, None)

        sph.sphn(names, exclude_self=True)

        vals = [4., 4., 4., 4., 4., 4., 4., 4., 8.]
        
        for i in range(4):
            self.assertEqual(check_array(vals, arrs[i].get_npy_array()), True)        

        vals = [5., 5., 5., 5., 5., 5., 5., 5., 9.]

        for i in range(4):
            arrs[i].get_npy_array()[:] = 0.0

        sph.sphn(names, exclude_self=False)

        for i in range(4):
            self.assertEqual(check_array(vals, arrs[i].get_npy_array()), True)
        
if __name__ == '__main__':
    unittest.main()
