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


    parr1 = ParticleArray(name='parr1', **{'x':{'data':x}, 'y':{'data':y},
                                           'z':{'data':z}, 'h':{'data':h},
                                           'm':{'data':m},
                                           'rho':{'data':rho},
                                           'velx':{'data':u},
                                           'v':{'data':v},
                                           'w':{'data':w}})

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
    
    parr1 = ParticleArray(name='parr1', **{'x':{'data':x},
                                                                  'y':{'data':y},
                                                                  'z':{'data':z},
                                                                  'm':{'data':m},
                                                                  'rho':{'data':rho},
                                                                  'h':{'data':h},
                                                                  't':{'data':t}})

    return [parr1]
