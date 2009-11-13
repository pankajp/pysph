"""
Tests for the density_funcs module.
"""

# standard imports
import unittest
import numpy

# local imports
from pysph.sph.density_funcs import SPHRho3D
from pysph.base.particle_array import ParticleArray
from pysph.base.kernel3d import Poly6Kernel3D, CubicSpline3D

def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-16."""
    return numpy.allclose(x, y, atol=1e-16, rtol=0)

def get_sample_data1(mass=1.0, radius=1.0):
    """
    Generate sample data for tests in this module.

    A particle array with two points (0, 0, 0) and (1, 1, 1) is created and
    returned. 
    """
    x = [0.0, 1.0]
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    m = [mass, mass]
    h = [radius, radius]
    
    p = ParticleArray(x={'data':x}, y={'data':y}, z={'data':z},
                      m={'data':m}, h={'data':h})
    return p

def get_sample_data2(mass=1.0, radius=1.0):
    """
    Generate sample data for tests in this module.

    Two particle arrays with one point each is returned. The points and values
    at them are the same as in the above function.
    """
    p1 = ParticleArray(x={'data':[0.0]},
                       y={'data':[0.0]},
                       z={'data':[0.0]},
                       m={'data':[mass]},
                       h={'data':[radius]})

    p2 = ParticleArray(x={'data':[1.0]},
                       y={'data':[1.0]},
                       z={'data':[1.0]},
                       m={'data':[mass]},
                       h={'data':[radius]})

    return p1, p2
    
class TestSPHRho3D(unittest.TestCase):
    """
    Tests for the SPHRho3D class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        parr = get_sample_data1()
        sph_rho3d = SPHRho3D(parr, parr)
        
        self.assertEqual(sph_rho3d.source, parr)
        self.assertEqual(sph_rho3d.dest, parr)
        self.assertEqual(sph_rho3d.output_fields(), 1)


        parr1, parr2 = get_sample_data2()
        sph_rho3d = SPHRho3D(parr1, parr2)

        self.assertEqual(sph_rho3d.source, parr1)
        self.assertEqual(sph_rho3d.dest, parr2)
        self.assertEqual(sph_rho3d.output_fields(), 1)

    def test_eval_1(self):
        """
        Tests the eval function.
        """
        parr = get_sample_data1()
        sph_rho3d = SPHRho3D(parr, parr)
        
        nr = numpy.array([0.])
        dnr = numpy.array([0.])
        
        k = Poly6Kernel3D()
        # get contribution of particle 1 on particle 0
        sph_rho3d.py_eval(1, 0, k, nr, dnr)

        self.assertEqual(check_array(nr, [0]), True)
        self.assertEqual(check_array(nr, [0]), True)

        sph_rho3d.py_eval(0, 1, k, nr, dnr)

        self.assertEqual(check_array(nr, [0]), True)
        self.assertEqual(check_array(dnr, [0]), True)

        nr[0] = 0
        dnr[0] = 0
        sph_rho3d.py_eval(0, 0, k, nr, dnr)
        self.assertEqual(check_array(nr, [1.5666814710608448]), True)
        self.assertEqual(check_array(dnr, [0]), True)

        nr[0] = 0
        dnr[0] = 0
        sph_rho3d.py_eval(1, 1, k, nr, dnr)
        
        self.assertEqual(check_array(nr, [1.5666814710608448]), True)
        self.assertEqual(check_array(dnr, [0]), True)

    def test_eval_2(self):
        """
        Tests the eval function.
        """
        parr1, parr2 = get_sample_data2(mass=4.0, radius=2.0)
        
        sph_rho3d = SPHRho3D(parr1, parr2)
        
        nr = numpy.array([0.])
        dnr = numpy.array([0.])
        
        k = Poly6Kernel3D()
        # get contribution of particle 1 on particle 0
        nr[0] = 4.0
        sph_rho3d.py_eval(0, 0, k, nr, dnr)

        self.assertEqual(check_array(nr, [4.01223969899266285]), True)
        self.assertEqual(check_array(dnr, [0]), True)

if __name__ == '__main__':
    unittest.main()
