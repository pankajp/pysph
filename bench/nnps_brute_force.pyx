cimport numpy as npy
import numpy

from libcpp.vector cimport vector

import time

cdef inline double square(double dx, double dy, double dz):
    return dx*dx + dy*dy + dz*dz

cdef list Ns = [10000, 20000, 40000, 80000, 160000]

def get_points(np = 1000000):
    x = numpy.random.random(np)*2.0 - 1.0
    y = numpy.random.random(np)*2.0 - 1.0

    return x, y

cpdef brute_force_nnps(long np, double radius):
    """
    Brute force computation of nearest neighbors using an N^2
    algorithm.

    Parameters:
    -----------

    np -- the size of the array

    xar -- the x array
    yar -- the y array

    """

    cdef npy.ndarray[npy.float64_t, ndim=1] xa = \
         numpy.random.random(np)*2.0 - 1.0

    cdef npy.ndarray[npy.float64_t, ndim=1] ya = \
         numpy.random.random(np)*2.0 - 1.0

    cdef vector[long] indices
    
    cdef double r2 = radius*radius
    cdef long i, j
    cdef double dist

    cdef double xi, yi, xj, yj

    for i in range(np):
        xi = xa[i]
        yi = ya[i]

        indices.clear()

        for j in range(np):
            xj = xa[j]
            yj = ya[j]

            dist = square(xi-xj, yi-yj, 0.0)

            if dist <= r2:
                indices.push_back(i)


cpdef get_neighbors():
    cdef dict ret = {}

    for np in Ns:

        # h ~ 2*vol_per_particle
        # rad ~ (2-3)*h => rad ~ 6*h
        
        vol_per_particle = pow(2.0/np, 0.5)
        radius = 6 * vol_per_particle
        
        t = time.time()
        brute_force_nnps(np, radius)
        t = time.time() - t

        ret['Brute force for %d particles'%(np)] = t

    return ret

cdef list funcs = [get_neighbors]

cpdef bench():
    """returns a list of a dict of point operations timings"""
    cdef list timings = []
    for func in funcs:
        timings.append(func())
    return timings

if __name__ == '__main__':
    print bench()
