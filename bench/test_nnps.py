""" Benchmark for the PySPH neighbor search functions. """

import sys
import numpy
from random import random

#PySPH imports
import pysph.base.api as base

def get_particle_array(np = 10000):
    """ Get np particles in domain [1, -1] X [-1, 1]  """

    x = numpy.zeros(np, 'float')
    y = numpy.zeros(np, 'float')
    z = numpy.zeros(np, 'float')
    
    for i in range(np):
        r1 = random()
        r2 = random()
        
        sign1 = 1
        sign2 = 1
        
        if r1 > 0.5:
            sign1 = -1
        if r2 > 0.5:
            sign2 = -1

        x[i] = sign1*random()
        y[i] = sign2*random()
        
    h = numpy.ones_like(x) * 0.1
    
    pdict = {}
    pdict['x'] = {'name':'x', 'data':x}
    pdict['y'] = {'name':'y', 'data':y}
    pdict['z'] = {'name':'z', 'data':z}
    pdict['h'] = {'name':'h', 'data':h}
    
    pa = base.ParticleArray(**pdict)
    
    return pa

def bin_particles(pa, min_cell_size=0.1):
    """ Bin the particles. 
    
    Parameters:
    -----------
    
    pa -- a newly created particle array from the get_particle_array function
    min_cell_size -- the cell size to use for binning
    
    """
    
    particles = base.Particles([pa,], min_cell_size=min_cell_size)
    return particles

def cache_neighbors(pa):
    """ Cache the neighbors for the particle array """
    loc = particles.get_neighbor_particle_locator(pa,pa,2.0)
    loc.py_get_nearest_particles(0)
    

if __name__ == '__main__':
    if len(sys.argv) > 1:
        pa = get_particle_array(np = int(sys.argv[-1]))
    else:
        pa = get_particle_array()
        
    print "Number of particles: ", pa.get_number_of_particles()

    particles = bin_particles(pa)
    cache_neighbors(pa)

        