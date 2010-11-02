"""
Tests for the density_funcs module.
"""

# standard imports
import unittest
import numpy

# local imports
from pysph.sph.funcs.density_funcs import SPHRho, SPHDensityRate
from pysph.base.particle_array import ParticleArray
from pysph.base.kernels import Poly6Kernel, CubicSplineKernel

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
    
class TestSPHRho(unittest.TestCase):
    """
    Tests for the SPHRho class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        parr = get_sample_data1()
        sph_rho = SPHRho(parr, parr)
        
        self.assertEqual(sph_rho.source, parr)
        self.assertEqual(sph_rho.dest, parr)


        parr1, parr2 = get_sample_data2()
        sph_rho = SPHRho(parr1, parr2)

        self.assertEqual(sph_rho.source, parr1)
        self.assertEqual(sph_rho.dest, parr2)

    def test_eval_1(self):
        """
        Tests the eval function.
        """
        parr = get_sample_data1()
        sph_rho = SPHRho(parr, parr)
        
        k = Poly6Kernel()
        # get contribution of particle 1 on particle 0
        nr, dnr = sph_rho.py_eval(1, 0, k)
        self.assertEqual(check_array(nr, [0]), True)
        self.assertEqual(check_array(nr, [0]), True)

        nr, dnr = sph_rho.py_eval(0, 1, k)
        self.assertEqual(check_array(nr, [0]), True)
        self.assertEqual(check_array(dnr, [0]), True)

        nr, dnr = sph_rho.py_eval(0, 0, k)
        self.assertEqual(check_array(nr, [1.5666814710608448]), True)
        self.assertEqual(check_array(dnr, [0]), True)

        nr, dnr = sph_rho.py_eval(1, 1, k)
        self.assertEqual(check_array(nr, [1.5666814710608448]), True)
        self.assertEqual(check_array(dnr, [0]), True)

    def test_eval_2(self):
        """
        Tests the eval function.
        """
        parr1, parr2 = get_sample_data2(mass=4.0, radius=2.0)
        
        sph_rho = SPHRho(parr1, parr2)
        
        k = Poly6Kernel()
        # get contribution of particle 1 on particle 0
        nr, dnr = sph_rho.py_eval(0, 0, k)
        self.assertEqual(check_array(nr, [0.01223969899266285]), True)
        self.assertEqual(check_array(dnr, [0]), True)

if __name__ == '__main__':
    unittest.main()
