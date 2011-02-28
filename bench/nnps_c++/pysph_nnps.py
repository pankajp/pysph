""" Benchmark for the PySPH neighbor search functions. """
import sys
import numpy
import time

#PySPH imports
import pysph.base.api as base

def get_points(np = 10000):
    """ Get np particles in domain [1, -1] X [-1, 1]  """

    x = numpy.random.random(np)*2.0 - 1.0
    y = numpy.random.random(np)*2.0 - 1.0
    z = numpy.random.random(np)*2.0 - 1.0

    # h ~ 2*vol_per_particle
    # rad ~ (2-3)*h => rad ~ 6*h
        
    vol_per_particle = pow(4.0/np, 0.5)
    radius = 6 * vol_per_particle

    h = numpy.ones_like(x) * radius * 0.5

    return x, y, z, h
   
def get_particle_array(x, y, z, h):
    pdict = {}
    pdict['x'] = {'name':'x', 'data':x}
    pdict['y'] = {'name':'y', 'data':y}
    pdict['z'] = {'name':'z', 'data':z}
    pdict['h'] = {'name':'h', 'data':h}
    
    pa = base.ParticleArray(**pdict)
    return pa

def bin_particles(pa):
    """ Bin the particles. 
    
    Parameters:
    -----------
    
    pa -- a newly created particle array from the get_particle_array function
    min_cell_size -- the cell size to use for binning
    
    """
    
    particles = base.Particles([pa,])
    return particles

def cache_neighbors(particles):
    """ Cache the neighbors for the particle array """
    pa = particles.arrays[0]
    loc = particles.get_neighbor_particle_locator(pa,pa,2.0)
    loc.py_get_nearest_particles(0)

def get_stats(particles):
    cd = particles.cell_manager.cells_dict

    ncells = len(cd)
    np_max = 0
    _np = 0

    for cid, cell in cd.iteritems():
        np = cell.index_lists[0].length
        _np += np
        if np > np_max:
            np_max = np

    
    print "\n\n\n##############################################################"
    print "CELL MANAGER DATA"
    print "CellManager cell size ", particles.cell_manager.cell_size
    print "Number of cells %d\t Particles/cell (avg) %f "%(ncells, _np/ncells),
    print " Maximum %d particles"%(np_max)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        np = sys.argv[-1]
        x,y,z,h = get_points(np = int(sys.argv[-1]))
        pa = get_particle_array(x,y,z,h)
    else:
        x,y,z,h = get_points()
        pa = get_particle_array(x,y,z,h)
        

    np = pa.get_number_of_particles()

    print "Number of particles: ", np

    vol_per_particle = pow(4.0/np, 0.5)
    radius = 6 * vol_per_particle

    print "Search Radius %f. "%(radius)

    t = time.time()
    particles = bin_particles(pa)
    t = time.time() - t

    print "Time for binning: %f s" %(t)

    t = time.time()
    cache_neighbors(particles)
    t = time.time() - t

    print "Time for caching neighbors: %f s" %(t)

    get_stats(particles)
