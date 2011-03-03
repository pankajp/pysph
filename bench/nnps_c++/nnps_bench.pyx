cimport cython

import time
import sys

cimport cython_nnps as nnps

import numpy as npy
cimport numpy as npy

def get_points(np=10000):
    x = npy.random.random(np)*2.0 - 1.0
    y = npy.random.random(np)*2.0 - 1.0
    z = npy.random.random(np)*2.0 - 1.0

    return x,y,z


@cython.boundscheck(False)
cdef cache_neighbors(npy.ndarray[npy.float64_t, ndim=1] x,
                     npy.ndarray[npy.float64_t, ndim=1] y,
                     npy.ndarray[npy.float64_t, ndim=1] z):

    cdef list nbr_cache = list()
    cdef list inbrs

    cdef int i
    cdef nnps.NNPS nps

    cdef long np = len(x)
    
    cdef double vol_per_particle = npy.power(8.0/np, 1.0/3.0)
    cdef double radius = 6 * vol_per_particle

    cdef double xi, yi, zi

    nps = nnps.NNPS(x,y,z)
    
    nps.set_cell_sizes(radius)

    t = time.time()
    nps.bin_particles()
    t = time.time() - t

    print "Time for binning %d particles: %fs"%(np, t)
    
    t = time.time()
    for i in range(np):
        inbrs = list()

        xi = x[i]; yi = y[i]; zi = z[i]

        nps.c_get_nearest_particles(xi, yi, zi, radius, inbrs)
        nbr_cache.append(inbrs)

    t = time.time() - t

    print "Time to cache %d particles: %fs"%(np, t)

cpdef run_test(np=10000):
    x,y,z = get_points(np)

    cache_neighbors(x,y,z)

if __name__ == '__main__':
    np = 10000

    if len(sys.argv) > 1:
        np = int(sys.argv[-1])

    run_test(np)
        
