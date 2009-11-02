# standard imports
import numpy

# local imports
from pysph.base.particle_array import *


def generate_sample_dataset_1():
    """
    Generate test test data.

    Look at image test_cell_case1.png for details.
    """
    x = numpy.array([0.25, 0.8, 0.5, 0.8, 0.2, 0.5, 1.5, 1.5])
    y = numpy.array([0.25, 0.1, 0.5, 0.8, 0.9, 1.5, 0.5, 1.5])
    z = numpy.array([0., 0, 0, 0, 0, 0, 0, 0])
    h = numpy.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    parr1 = ParticleArray(particle_manager=None, name='parr1', **{'x':x, 'y':y, 'z':z, 'h':h})

    x = numpy.array([0.2, 1.2, 1.5, 0.4])
    y = numpy.array([0., 0, 0, 0])
    z = numpy.array([1.6, 1.5, -0.5, 0.4])
    h = numpy.array([1.0, 1.0, 1.0, 1.0])

    parr2 = ParticleArray(particle_manager=None, name='parr2', **{'x':x, 'y':y, 'z':z, 'h':h})

    return [parr1, parr2]

def generate_sample_dataset_2():
    """
    Generate test test data.
    
    Look at image test_cell_data2.png for details.
    """
    x = numpy.array([-0.5, -0.5, 0.5, 0.5, 1.5, 2.5, 2.5])
    y = numpy.array([2.5, -0.5, 1.5, 0.5, 0.5, 0.5, -0.5])
    z = numpy.array([0., 0, 0, 0, 0, 0, 0])
    h = numpy.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    parr1 = ParticleArray(particle_manager=None, name='parr1', **{'x':x, 'y':y, 'z':z, 'h':h})

    return [parr1]
    
