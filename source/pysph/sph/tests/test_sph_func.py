"""
Test the base classes from the sph_func module.
"""

# standard imports
import unittest
import numpy

# local imports
from pysph.sph.sph_func import *
from pysph.sph.tests.common_data import *
from pysph.base.point import Point
from pysph.base.carray import DoubleArray

class TestMakeCoords(unittest.TestCase):
    """
    Tests the make_coords_*d functions.
    """
    def test(self):
        x = DoubleArray(2)
        x.set_data(numpy.array([1.0, 3.0]))
    
        p = Point(-1, -1, -1)
        
        py_make_coords_1d(x, p, 0)

        self.assertEqual(p.x, 1.0)
        self.assertEqual(p.y, 0.0)
        self.assertEqual(p.z, 0.0)

        py_make_coords_1d(x, p, 1)
        self.assertEqual(p.x, 3.0)
        self.assertEqual(p.y, 0.0)
        self.assertEqual(p.z, 0.0)

        p = Point(-1, -1, -1)

        y = DoubleArray(2)
        y.set_data(numpy.array([-12. , 5.0]))
        
        py_make_coords_2d(x, y, p, 0)
        self.assertEqual(p.x, 1.0)
        self.assertEqual(p.y, -12)
        self.assertEqual(p.z, 0.0)

        py_make_coords_2d(x, y, p, 1)
        self.assertEqual(p.x, 3.0)
        self.assertEqual(p.y, 5.0)
        self.assertEqual(p.z, 0.0)

        p = Point(-1, -1, -1)
        z = DoubleArray(2)
        z.set_data(numpy.array([-4, 0.1]))
        py_make_coords_3d(x, y, z, p, 0)
        self.assertEqual(p.x, 1.0)
        self.assertEqual(p.y, -12)
        self.assertEqual(p.z, -4.0)

        py_make_coords_3d(x, y, z, p, 1)
        self.assertEqual(p.x, 3.0)
        self.assertEqual(p.y, 5.0)
        self.assertEqual(p.z, 0.1)
        
################################################################################
# `TestSPHFunctionParticle` class.
################################################################################
class TestSPHFunctionParticle(unittest.TestCase):
    """
    Tests the SPHFunctionParticle class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        parrs= generate_sample_dataset_1()
        
        f = SPHFunctionParticle(parrs[0], parrs[0], setup_arrays=False)

        self.assertEqual(f.dest, parrs[0])
        self.assertEqual(f.source, parrs[0])
        
        self.assertEqual(f.h, 'h')
        self.assertEqual(f.mass, 'm')
        self.assertEqual(f.rho, 'rho')

        self.assertEqual(f.s_h, None)
        self.assertEqual(f.s_mass, None)
        self.assertEqual(f.s_rho, None)
        
        self.assertEqual(f.d_h, None)
        self.assertEqual(f.d_mass, None)
        self.assertEqual(f.d_rho, None)

        f = SPHFunctionParticle(parrs[0], parrs[0], h='h',
                                mass='m', rho='rho')

        self.assertEqual(f.dest, parrs[0])
        self.assertEqual(f.source, parrs[0])
        
        self.assertEqual(f.h, 'h')
        self.assertEqual(f.mass, 'm')
        self.assertEqual(f.rho, 'rho')

        # check that the arrays have been setup.
        self.assertEqual(f.s_h, parrs[0].get_carray('h'))
        self.assertEqual(f.s_mass, parrs[0].get_carray('m'))
        self.assertEqual(f.s_rho, parrs[0].get_carray('rho'))
        self.assertEqual(f.d_h, parrs[0].get_carray('h'))
        self.assertEqual(f.d_mass, parrs[0].get_carray('m'))
        self.assertEqual(f.d_rho, parrs[0].get_carray('rho'))

    def test_output_fields(self):
        """
        Tests the output_fields function.
        """
        parrs= generate_sample_dataset_1()
        f = SPHFunctionParticle(parrs[0], parrs[0])

        self.assertRaises(NotImplementedError, f.output_fields)

class TestSPHFunctionParticle1D(unittest.TestCase):
    """
    Tests the SPHFunctionParticle3D class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        parrs = generate_sample_dataset_1()

        f = SPHFunctionParticle1D(parrs[0], parrs[0], setup_arrays=False)
        
        self.assertEqual(f.dest, parrs[0])
        self.assertEqual(f.source, parrs[0])
        
        self.assertEqual(f.h, 'h')
        self.assertEqual(f.mass, 'm')
        self.assertEqual(f.rho, 'rho')
        self.assertEqual(f.coord_x, 'x')
        self.assertEqual(f.velx, 'u')

        self.assertEqual(f.s_h, None)
        self.assertEqual(f.s_mass, None)
        self.assertEqual(f.s_rho, None)
        self.assertEqual(f.s_x, None)
        self.assertEqual(f.s_velx, None)
        
        self.assertEqual(f.d_h, None)
        self.assertEqual(f.d_mass, None)
        self.assertEqual(f.d_rho, None)
        self.assertEqual(f.d_x, None)
        self.assertEqual(f.d_velx, None)

        f = SPHFunctionParticle1D(parrs[0], parrs[0], h='h',
                                mass='m', rho='rho', velx='velx')

        self.assertEqual(f.dest, parrs[0])
        self.assertEqual(f.source, parrs[0])
        
        self.assertEqual(f.h, 'h')
        self.assertEqual(f.mass, 'm')
        self.assertEqual(f.rho, 'rho')

        # check that the arrays have been setup.
        self.assertEqual(f.s_h, parrs[0].get_carray('h'))
        self.assertEqual(f.s_mass, parrs[0].get_carray('m'))
        self.assertEqual(f.s_rho, parrs[0].get_carray('rho'))
        self.assertEqual(f.d_h, parrs[0].get_carray('h'))
        self.assertEqual(f.d_mass, parrs[0].get_carray('m'))
        self.assertEqual(f.d_rho, parrs[0].get_carray('rho'))

        self.assertEqual(f.s_x, parrs[0].get_carray('x'))
        self.assertEqual(f.s_velx, parrs[0].get_carray('velx'))
        
        self.assertEqual(f.d_x, parrs[0].get_carray('x'))
        self.assertEqual(f.d_velx, parrs[0].get_carray('velx'))

    def test_output_fields(self):
        """
        Tests the output_fields function.
        """
        parrs= generate_sample_dataset_1()
        f = SPHFunctionParticle1D(parrs[0], parrs[0])

        self.assertRaises(NotImplementedError, f.output_fields)

################################################################################
# `TestSPHFunctionParticle2D` class.
################################################################################
class TestSPHFunctionParticle2D(unittest.TestCase):
    """
    Tests the SPHFunctionParticle2D class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        parrs = generate_sample_dataset_1()

        f = SPHFunctionParticle2D(parrs[0], parrs[0], setup_arrays=False)
        
        self.assertEqual(f.dest, parrs[0])
        self.assertEqual(f.source, parrs[0])
        
        self.assertEqual(f.h, 'h')
        self.assertEqual(f.mass, 'm')
        self.assertEqual(f.rho, 'rho')
        self.assertEqual(f.coord_x, 'x')
        self.assertEqual(f.coord_y, 'y')
        self.assertEqual(f.velx, 'u')
        self.assertEqual(f.vely, 'v')

        self.assertEqual(f.s_h, None)
        self.assertEqual(f.s_mass, None)
        self.assertEqual(f.s_rho, None)
        self.assertEqual(f.s_x, None)
        self.assertEqual(f.s_velx, None)
        self.assertEqual(f.s_y, None)
        self.assertEqual(f.s_vely, None)
        
        self.assertEqual(f.d_h, None)
        self.assertEqual(f.d_mass, None)
        self.assertEqual(f.d_rho, None)
        self.assertEqual(f.d_x, None)
        self.assertEqual(f.d_velx, None)
        self.assertEqual(f.d_y, None)
        self.assertEqual(f.d_vely, None)

        f = SPHFunctionParticle2D(parrs[0], parrs[0], h='h',
                                mass='m', rho='rho', velx='velx')

        self.assertEqual(f.dest, parrs[0])
        self.assertEqual(f.source, parrs[0])
        
        self.assertEqual(f.h, 'h')
        self.assertEqual(f.mass, 'm')
        self.assertEqual(f.rho, 'rho')

        # check that the arrays have been setup.
        self.assertEqual(f.s_h, parrs[0].get_carray('h'))
        self.assertEqual(f.s_mass, parrs[0].get_carray('m'))
        self.assertEqual(f.s_rho, parrs[0].get_carray('rho'))
        self.assertEqual(f.d_h, parrs[0].get_carray('h'))
        self.assertEqual(f.d_mass, parrs[0].get_carray('m'))
        self.assertEqual(f.d_rho, parrs[0].get_carray('rho'))

        self.assertEqual(f.s_x, parrs[0].get_carray('x'))
        self.assertEqual(f.s_velx, parrs[0].get_carray('velx'))
        self.assertEqual(f.s_y, parrs[0].get_carray('y'))
        self.assertEqual(f.s_vely, parrs[0].get_carray('v'))
        
        self.assertEqual(f.d_x, parrs[0].get_carray('x'))
        self.assertEqual(f.d_velx, parrs[0].get_carray('velx'))
        self.assertEqual(f.d_y, parrs[0].get_carray('y'))
        self.assertEqual(f.d_vely, parrs[0].get_carray('v'))

    def test_output_fields(self):
        """
        Tests the output_fields function.
        """
        parrs= generate_sample_dataset_1()
        f = SPHFunctionParticle2D(parrs[0], parrs[0])

        self.assertRaises(NotImplementedError, f.output_fields)

################################################################################
# `TestSPHFunctionParticle3D` class.
################################################################################
class TestSPHFunctionParticle3D(unittest.TestCase):
    """
    Tests the SPHFunctionParticle3D class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        parrs = generate_sample_dataset_1()

        f = SPHFunctionParticle3D(parrs[0], parrs[0], setup_arrays=False)
        
        self.assertEqual(f.dest, parrs[0])
        self.assertEqual(f.source, parrs[0])
        
        self.assertEqual(f.h, 'h')
        self.assertEqual(f.mass, 'm')
        self.assertEqual(f.rho, 'rho')
        self.assertEqual(f.coord_x, 'x')
        self.assertEqual(f.coord_y, 'y')
        self.assertEqual(f.coord_z, 'z')
        self.assertEqual(f.velx, 'u')
        self.assertEqual(f.vely, 'v')
        self.assertEqual(f.velz, 'w')

        self.assertEqual(f.s_h, None)
        self.assertEqual(f.s_mass, None)
        self.assertEqual(f.s_rho, None)
        self.assertEqual(f.s_x, None)
        self.assertEqual(f.s_velx, None)
        self.assertEqual(f.s_y, None)
        self.assertEqual(f.s_vely, None)
        self.assertEqual(f.s_z, None)
        self.assertEqual(f.s_velz, None)
 
        self.assertEqual(f.d_h, None)
        self.assertEqual(f.d_mass, None)
        self.assertEqual(f.d_rho, None)
        self.assertEqual(f.d_x, None)
        self.assertEqual(f.d_velx, None)
        self.assertEqual(f.d_y, None)
        self.assertEqual(f.d_vely, None)
        self.assertEqual(f.d_z, None)
        self.assertEqual(f.d_velz, None)


        f = SPHFunctionParticle3D(parrs[0], parrs[0], h='h',
                                mass='m', rho='rho', velx='velx')

        self.assertEqual(f.dest, parrs[0])
        self.assertEqual(f.source, parrs[0])
        
        self.assertEqual(f.h, 'h')
        self.assertEqual(f.mass, 'm')
        self.assertEqual(f.rho, 'rho')

        # check that the arrays have been setup.
        self.assertEqual(f.s_h, parrs[0].get_carray('h'))
        self.assertEqual(f.s_mass, parrs[0].get_carray('m'))
        self.assertEqual(f.s_rho, parrs[0].get_carray('rho'))
        self.assertEqual(f.d_h, parrs[0].get_carray('h'))
        self.assertEqual(f.d_mass, parrs[0].get_carray('m'))
        self.assertEqual(f.d_rho, parrs[0].get_carray('rho'))

        self.assertEqual(f.s_x, parrs[0].get_carray('x'))
        self.assertEqual(f.s_velx, parrs[0].get_carray('velx'))
        self.assertEqual(f.s_y, parrs[0].get_carray('y'))
        self.assertEqual(f.s_vely, parrs[0].get_carray('v'))
        self.assertEqual(f.s_z, parrs[0].get_carray('z'))
        self.assertEqual(f.s_velz, parrs[0].get_carray('w'))
        
        self.assertEqual(f.d_x, parrs[0].get_carray('x'))
        self.assertEqual(f.d_velx, parrs[0].get_carray('velx'))
        self.assertEqual(f.d_y, parrs[0].get_carray('y'))
        self.assertEqual(f.d_vely, parrs[0].get_carray('v'))
        self.assertEqual(f.d_z, parrs[0].get_carray('z'))
        self.assertEqual(f.d_velz, parrs[0].get_carray('w'))

    def test_output_fields(self):
        """
        Tests the output_fields function.
        """
        parrs= generate_sample_dataset_1()
        f = SPHFunctionParticle3D(parrs[0], parrs[0])

        self.assertRaises(NotImplementedError, f.output_fields)


################################################################################
# `TestSPHFunctionPoint` class.
################################################################################
class TestSPHFunctionPoint(unittest.TestCase):
    """
    Tests the SPHFunctionPoint class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        parrs= generate_sample_dataset_1()
        
        f = SPHFunctionPoint(parrs[0], setup_arrays=False)

        self.assertEqual(f.source, parrs[0])
        
        self.assertEqual(f.h, 'h')
        self.assertEqual(f.mass, 'm')
        self.assertEqual(f.rho, 'rho')

        self.assertEqual(f.s_h, None)
        self.assertEqual(f.s_mass, None)
        self.assertEqual(f.s_rho, None)

        f = SPHFunctionPoint(parrs[0], h='h',
                                mass='m', rho='rho')

        self.assertEqual(f.source, parrs[0])
        
        self.assertEqual(f.h, 'h')
        self.assertEqual(f.mass, 'm')
        self.assertEqual(f.rho, 'rho')

        # check that the arrays have been setup.
        self.assertEqual(f.s_h, parrs[0].get_carray('h'))
        self.assertEqual(f.s_mass, parrs[0].get_carray('m'))
        self.assertEqual(f.s_rho, parrs[0].get_carray('rho'))

    def test_output_fields(self):
        """
        Tests the output_fields function.
        """
        parrs= generate_sample_dataset_1()
        f = SPHFunctionPoint(parrs[0])

        self.assertRaises(NotImplementedError, f.output_fields)


################################################################################
# `TestSPHFunctionPoint1D` class.
################################################################################
class TestSPHFunctionPoint1D(unittest.TestCase):
    """
    Tests the SPHFunctionPoint1D class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        parrs = generate_sample_dataset_1()

        f = SPHFunctionPoint1D(parrs[0], setup_arrays=False)
        
        self.assertEqual(f.source, parrs[0])
        
        self.assertEqual(f.h, 'h')
        self.assertEqual(f.mass, 'm')
        self.assertEqual(f.rho, 'rho')
        self.assertEqual(f.coord_x, 'x')
        self.assertEqual(f.velx, 'u')

        self.assertEqual(f.s_h, None)
        self.assertEqual(f.s_mass, None)
        self.assertEqual(f.s_rho, None)
        self.assertEqual(f.s_x, None)
        self.assertEqual(f.s_velx, None)

        f = SPHFunctionPoint1D(parrs[0], h='h',
                               mass='m', rho='rho', velx='velx')
        self.assertEqual(f.source, parrs[0])
        
        self.assertEqual(f.h, 'h')
        self.assertEqual(f.mass, 'm')
        self.assertEqual(f.rho, 'rho')

        # check that the arrays have been setup.
        self.assertEqual(f.s_h, parrs[0].get_carray('h'))
        self.assertEqual(f.s_mass, parrs[0].get_carray('m'))
        self.assertEqual(f.s_rho, parrs[0].get_carray('rho'))

        self.assertEqual(f.s_x, parrs[0].get_carray('x'))
        self.assertEqual(f.s_velx, parrs[0].get_carray('velx'))

    def test_output_fields(self):
        """
        Tests the output_fields function.
        """
        parrs= generate_sample_dataset_1()
        f = SPHFunctionPoint1D(parrs[0])

        self.assertRaises(NotImplementedError, f.output_fields)

################################################################################
# `TestSPHFunctionPoint2D` class.
################################################################################
class TestSPHFunctionPoint2D(unittest.TestCase):
    """
    Tests the SPHFunctionPoint2D class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        parrs = generate_sample_dataset_1()

        f = SPHFunctionPoint2D(parrs[0], setup_arrays=False)
        
        self.assertEqual(f.source, parrs[0])
        
        self.assertEqual(f.h, 'h')
        self.assertEqual(f.mass, 'm')
        self.assertEqual(f.rho, 'rho')
        self.assertEqual(f.coord_x, 'x')
        self.assertEqual(f.coord_y, 'y')
        self.assertEqual(f.velx, 'u')
        self.assertEqual(f.vely, 'v')

        self.assertEqual(f.s_h, None)
        self.assertEqual(f.s_mass, None)
        self.assertEqual(f.s_rho, None)
        self.assertEqual(f.s_x, None)
        self.assertEqual(f.s_velx, None)
        self.assertEqual(f.s_y, None)
        self.assertEqual(f.s_vely, None)
        
        f = SPHFunctionPoint2D(parrs[0], h='h',
                               mass='m', rho='rho', velx='velx')
        self.assertEqual(f.source, parrs[0])
        
        self.assertEqual(f.h, 'h')
        self.assertEqual(f.mass, 'm')
        self.assertEqual(f.rho, 'rho')

        # check that the arrays have been setup.
        self.assertEqual(f.s_h, parrs[0].get_carray('h'))
        self.assertEqual(f.s_mass, parrs[0].get_carray('m'))
        self.assertEqual(f.s_rho, parrs[0].get_carray('rho'))
        self.assertEqual(f.s_x, parrs[0].get_carray('x'))
        self.assertEqual(f.s_velx, parrs[0].get_carray('velx'))
        self.assertEqual(f.s_y, parrs[0].get_carray('y'))
        self.assertEqual(f.s_vely, parrs[0].get_carray('v'))
        
    def test_output_fields(self):
        """
        Tests the output_fields function.
        """
        parrs= generate_sample_dataset_1()
        f = SPHFunctionPoint2D(parrs[0])

        self.assertRaises(NotImplementedError, f.output_fields)

################################################################################
# `TestSPHFunctionPoint3D` class.
################################################################################
class TestSPHFunctionPoint3D(unittest.TestCase):
    """
    Tests the SPHFunctionPoint3D class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        parrs = generate_sample_dataset_1()

        f = SPHFunctionPoint3D(parrs[0], setup_arrays=False)
        self.assertEqual(f.source, parrs[0])
        
        self.assertEqual(f.h, 'h')
        self.assertEqual(f.mass, 'm')
        self.assertEqual(f.rho, 'rho')
        self.assertEqual(f.coord_x, 'x')
        self.assertEqual(f.coord_y, 'y')
        self.assertEqual(f.coord_z, 'z')
        self.assertEqual(f.velx, 'u')
        self.assertEqual(f.vely, 'v')
        self.assertEqual(f.velz, 'w')

        self.assertEqual(f.s_h, None)
        self.assertEqual(f.s_mass, None)
        self.assertEqual(f.s_rho, None)
        self.assertEqual(f.s_x, None)
        self.assertEqual(f.s_velx, None)
        self.assertEqual(f.s_y, None)
        self.assertEqual(f.s_vely, None)
        self.assertEqual(f.s_z, None)
        self.assertEqual(f.s_velz, None)
 
        f = SPHFunctionPoint3D(parrs[0], h='h',
                               mass='m', rho='rho', velx='velx')
        self.assertEqual(f.source, parrs[0])
        
        self.assertEqual(f.h, 'h')
        self.assertEqual(f.mass, 'm')
        self.assertEqual(f.rho, 'rho')

        # check that the arrays have been setup.
        self.assertEqual(f.s_h, parrs[0].get_carray('h'))
        self.assertEqual(f.s_mass, parrs[0].get_carray('m'))
        self.assertEqual(f.s_rho, parrs[0].get_carray('rho'))
        self.assertEqual(f.s_x, parrs[0].get_carray('x'))
        self.assertEqual(f.s_velx, parrs[0].get_carray('velx'))
        self.assertEqual(f.s_y, parrs[0].get_carray('y'))
        self.assertEqual(f.s_vely, parrs[0].get_carray('v'))
        self.assertEqual(f.s_z, parrs[0].get_carray('z'))
        self.assertEqual(f.s_velz, parrs[0].get_carray('w'))

    def test_output_fields(self):
        """
        Tests the output_fields function.
        """
        parrs= generate_sample_dataset_1()
        f = SPHFunctionPoint3D(parrs[0])

        self.assertRaises(NotImplementedError, f.output_fields)
    
if __name__ == '__main__':
    unittest.main()
