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
    
if __name__ == '__main__':
    unittest.main()
