"""
Test the base classes from the sph_func module.
"""

# standard imports
import unittest
import numpy

# local imports
from pysph.sph.sph_func import SPHFunctionPoint, SPHFunctionParticle
from pysph.sph.tests.common_data import *
from pysph.base.point import Point
from pysph.base.carray import DoubleArray

###############################################################################
# `TestSPHFunctionParticle` class.
###############################################################################
class TestSPHFunctionParticle(unittest.TestCase):
    """
    Tests the SPHFunctionParticle3D class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        parrs = generate_sample_dataset_1()

        f = SPHFunctionParticle(parrs[0], parrs[0], setup_arrays=False)
        
        self.assertEqual(f.dest, parrs[0])
        self.assertEqual(f.source, parrs[0])
        
        self.assertEqual(f.h, 'h')
        self.assertEqual(f.m, 'm')
        self.assertEqual(f.rho, 'rho')
        self.assertEqual(f.x, 'x')
        self.assertEqual(f.y, 'y')
        self.assertEqual(f.z, 'z')
        self.assertEqual(f.u, 'u')
        self.assertEqual(f.v, 'v')
        self.assertEqual(f.w, 'w')

        self.assertEqual(f.s_h, None)
        self.assertEqual(f.s_m, None)
        self.assertEqual(f.s_rho, None)
        self.assertEqual(f.s_x, None)
        self.assertEqual(f.s_u, None)
        self.assertEqual(f.s_y, None)
        self.assertEqual(f.s_v, None)
        self.assertEqual(f.s_z, None)
        self.assertEqual(f.s_w, None)
 
        self.assertEqual(f.d_h, None)
        self.assertEqual(f.d_m, None)
        self.assertEqual(f.d_rho, None)
        self.assertEqual(f.d_x, None)
        self.assertEqual(f.d_u, None)
        self.assertEqual(f.d_y, None)
        self.assertEqual(f.d_v, None)
        self.assertEqual(f.d_z, None)
        self.assertEqual(f.d_w, None)


        f = SPHFunctionParticle(parrs[0], parrs[0], h='h',
                                m='m', rho='rho', u='velx')
        f.u = 'velx'
        f.setup_arrays()
        self.assertEqual(f.dest, parrs[0])
        self.assertEqual(f.source, parrs[0])
        
        self.assertEqual(f.h, 'h')
        self.assertEqual(f.m, 'm')
        self.assertEqual(f.rho, 'rho')

        # check that the arrays have been setup.
        self.assertEqual(f.s_h, parrs[0].get_carray('h'))
        self.assertEqual(f.s_m, parrs[0].get_carray('m'))
        self.assertEqual(f.s_rho, parrs[0].get_carray('rho'))
        self.assertEqual(f.d_h, parrs[0].get_carray('h'))
        self.assertEqual(f.d_m, parrs[0].get_carray('m'))
        self.assertEqual(f.d_rho, parrs[0].get_carray('rho'))

        self.assertEqual(f.s_x, parrs[0].get_carray('x'))
        self.assertEqual(f.s_u, parrs[0].get_carray('velx'))
        self.assertEqual(f.s_y, parrs[0].get_carray('y'))
        self.assertEqual(f.s_v, parrs[0].get_carray('v'))
        self.assertEqual(f.s_z, parrs[0].get_carray('z'))
        self.assertEqual(f.s_w, parrs[0].get_carray('w'))
        
        self.assertEqual(f.d_x, parrs[0].get_carray('x'))
        self.assertEqual(f.d_u, parrs[0].get_carray('velx'))
        self.assertEqual(f.d_y, parrs[0].get_carray('y'))
        self.assertEqual(f.d_v, parrs[0].get_carray('v'))
        self.assertEqual(f.d_z, parrs[0].get_carray('z'))
        self.assertEqual(f.d_w, parrs[0].get_carray('w'))


###############################################################################
# `TestSPHFunctionPoint` class.
###############################################################################
class TestSPHFunctionPoint(unittest.TestCase):
    """
    Tests the SPHFunctionPoint class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        parrs = generate_sample_dataset_1()

        f = SPHFunctionPoint(parrs[0], setup_arrays=False)
        self.assertEqual(f.source, parrs[0])
        
        self.assertEqual(f.h, 'h')
        self.assertEqual(f.m, 'm')
        self.assertEqual(f.rho, 'rho')
        self.assertEqual(f.x, 'x')
        self.assertEqual(f.y, 'y')
        self.assertEqual(f.z, 'z')
        self.assertEqual(f.u, 'u')
        self.assertEqual(f.v, 'v')
        self.assertEqual(f.w, 'w')

        self.assertEqual(f.s_h, None)
        self.assertEqual(f.s_m, None)
        self.assertEqual(f.s_rho, None)
        self.assertEqual(f.s_x, None)
        self.assertEqual(f.s_u, None)
        self.assertEqual(f.s_y, None)
        self.assertEqual(f.s_v, None)
        self.assertEqual(f.s_z, None)
        self.assertEqual(f.s_w, None)
 
        f = SPHFunctionPoint(parrs[0], h='h',
                               m='m', rho='rho', u='velx')
        f.u = 'velx'
        f.setup_arrays()
        self.assertEqual(f.source, parrs[0])
        
        self.assertEqual(f.h, 'h')
        self.assertEqual(f.m, 'm')
        self.assertEqual(f.rho, 'rho')

        # check that the arrays have been setup.
        self.assertEqual(f.s_h, parrs[0].get_carray('h'))
        self.assertEqual(f.s_m, parrs[0].get_carray('m'))
        self.assertEqual(f.s_rho, parrs[0].get_carray('rho'))
        self.assertEqual(f.s_x, parrs[0].get_carray('x'))
        self.assertEqual(f.s_u, parrs[0].get_carray('velx'))
        self.assertEqual(f.s_y, parrs[0].get_carray('y'))
        self.assertEqual(f.s_v, parrs[0].get_carray('v'))
        self.assertEqual(f.s_z, parrs[0].get_carray('z'))
        self.assertEqual(f.s_w, parrs[0].get_carray('w'))

    def test_output_fields(self):
        """
        Tests the output_fields function.
        """
        parrs= generate_sample_dataset_1()
        f = SPHFunctionPoint(parrs[0])

        self.assertRaises(NotImplementedError, f.output_fields)
    
if __name__ == '__main__':
    unittest.main()
