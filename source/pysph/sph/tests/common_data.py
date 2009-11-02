"""
Module containing some data required for tests of the sph module.
"""
# standard imports
import numpy

# local imports
from pysph.base.particle_array import *

def generate_sample_dataset_1():
    """
    Generate test test data.
    Look at image sph_test_data1.png
    """
    x = numpy.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
    y = numpy.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    z = numpy.array([0., 0, 0, 0, 0, 0, 0, 0, 0])
    h = numpy.array([1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])
    m = numpy.array([1., 1, 1, 1, 1, 1, 1, 1, 1])
    rho = numpy.array([1., 1, 1, 1, 1, 1, 1, 1, 1])    
    u = numpy.zeros(9)
    v = numpy.zeros(9)
    w = numpy.zeros(9)


    parr1 = ParticleArray(particle_manager=None, name='parr1', **{'x':x, 'y':y,
                                                                  'z':z, 'h':h,
                                                                  'm':m,
                                                                  'rho':rho,
                                                                  'velx':u,
                                                                  'v':v,
                                                                  'w':w})

    return [parr1]



def generate_sample_dataset_2():
    """
    Generate test data.
    Look at image sph_test_data2.png.
    """
    x = numpy.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.5])
    y = numpy.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5])
    z = numpy.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.5])

    h = numpy.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    m = numpy.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    rho = numpy.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    t = numpy.zeros(9)
    
    parr1 = ParticleArray(particle_manager=None, name='parr1', **{'x':x,
                                                                  'y':y,
                                                                  'z':z,
                                                                  'm':m,
                                                                  'rho':rho,
                                                                  'h':h,
                                                                  't':t})

    return [parr1]
