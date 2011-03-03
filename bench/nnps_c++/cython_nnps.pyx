#cython: cdivision=True
""" Cython definitions for the Nearest Neighbor Particle Search algorithm """
cimport cython

import numpy
cimport numpy as np

cdef int IntPoint_maxint = 2**20

from cpython.dict cimport *
from cpython.list cimport *

cdef extern from 'limits.h':
    cdef double ceil(double)
    cdef double floor(double)
    cdef double fabs(double)

cdef class VoxelId:

    def __init__(self, int ix=0, int iy=0, int iz=0):
        self.ix = ix
        self.iy = iy
        self.iz = iz

    cdef set(self, int ix, int iy, int iz):
        self.ix = ix
        self.iy = iy
        self.iz = iz
       
    def __cmp__(self, VoxelId v): 

        if (self.ix < v.ix):
            return -1

        elif (self.ix > v.ix):
            return +1

        else:
            if ( self.iy < v.iy ):
                return -1

            elif (self.iy > v.iy):
                return +1

            else:
                if ( self.iz < v.iz ):
                    return -1
                elif ( self.iz > v.iz ):
                    return +1
                else:
                    return 0

    def __hash__(self):
        cdef long ret = self.ix + IntPoint_maxint
        ret = 2 * IntPoint_maxint * ret + self.iy + IntPoint_maxint

        return 2 * IntPoint_maxint * ret + self.iz + IntPoint_maxint
             
cdef class Voxel:

    def __init__(self, VoxelId vid):
        self.vid = vid
        self.indices = []

    def __cmp__(self, Voxel v):
        return ( self.vid < v )

    def __hash__(self):
        return hash( self.vid )

    def num_particles(self):
        return len(self.indices)

    def get_indices(self):
        return self.indices
    
cdef class NNPS:

    def __init__(self, x, y, z):

        msg = "Array sizes must be compatible "
        assert ( len(x) == len(y) ), msg
        assert ( len(y) == len(z) ), msg

        self.x = x
        self.y = y
        self.z = z

        self.np = len(x)

        self.Mx, self.mx = numpy.max(x), numpy.min(x)
        self.My, self.my = numpy.max(y), numpy.min(y)
        self.Mz, self.mz = numpy.max(z), numpy.min(z)

        self.voxels = {}

    def set_cell_sizes(self, double sx, double sy=0, double sz=0,
                       bint all_equal=True):
        self.sx = sx
        self.sy = sy
        self.sz = sz
        
        if all_equal:
            self.sy = sx
            self.sz = sx

    @cython.boundscheck(False)
    cdef c_bin_particles(self):

        cdef int i, ix, iy, iz

        cdef np.ndarray[np.float64_t, ndim=1] x = self.x
        cdef np.ndarray[np.float64_t, ndim=1] y = self.y
        cdef np.ndarray[np.float64_t, ndim=1] z = self.z

        cdef VoxelId vid
        cdef Voxel voxel

        PyDict_Clear(self.voxels)

        for i in range(self.np):
            ix = <int>floor( x[i]/self.sx )
            iy = <int>floor( y[i]/self.sy )
            iz = <int>floor( z[i]/self.sz )

            vid = VoxelId(ix, iy, iz)

            if PyDict_Contains( self.voxels, vid ):
                voxel = <Voxel>PyDict_GetItem( self.voxels,  vid )
            else:
                voxel = Voxel( vid )
                self.voxels[vid] = voxel

            PyList_Append(voxel.indices, i)

    def bin_particles(self):
        self.c_bin_particles()

    def get_sizes(self):
        return self.sx, self.sy, self.sz

    cdef c_get_adjacent_voxels(self, double xi, double yi, double zi,
                               list voxel_list, int distance=1 ):
        
        cdef int ix = <int>floor( xi/self.sx )
        cdef int iy = <int>floor( yi/self.sy )
        cdef int iz = <int>floor( zi/self.sz )

        cdef int i, j, k

        cdef int d = distance
        cdef VoxelId vid

        voxel_list[:] = []

        for i in range( ix-d, ix+d+1 ):
            for j in range ( iy-d, iy+d+1 ):
                for k in range ( iz-d, iz+d+1 ):
                    PyList_Append( voxel_list, VoxelId(i,j,k) )
                                    
    def get_adjacent_voxels(self, double xi, double yi, double zi,
                            int distance=1):
        cdef list voxel_list = []
        self.c_get_adjacent_voxels(xi, yi, zi, voxel_list, distance)
        
        return voxel_list

    cdef c_get_nearest_particles(self, double xi, double yi, double zi,
                                 double radius, list nbr_list):

        cdef voxel_list = []
        self.c_get_adjacent_voxels( xi, yi, zi, voxel_list )
        
        nbr_list[:] = []

        cdef np.ndarray[np.float64_t, ndim=1] x = self.x
        cdef np.ndarray[np.float64_t, ndim=1] y = self.y
        cdef np.ndarray[np.float64_t, ndim=1] z = self.z

        cdef double xj, yj, zj, dist

        cdef int nvoxels = PyList_Size(voxel_list)
        cdef VoxelId vid
        cdef Voxel voxel

        cdef int j, nindx, np
        cdef int i

        cdef list indices

        for i in range(nvoxels):
            vid = <VoxelId>PyList_GetItem(voxel_list, i)

            if PyDict_Contains(self.voxels, vid):
                voxel = <Voxel>PyDict_GetItem(self.voxels, vid)

                indices = voxel.indices

                np = PyList_Size(indices)

                for j in range( np ):
                    nindx = indices[j]

                    xj = x[nindx]
                    yj = y[nindx]
                    zj = z[nindx]
                    
                    dist = (xi-xj)*(xi-xj) + (yi-yj)*(yi-yj) + (zi-zj)*(zi-zj)

                    if dist < (radius*radius):
                        PyList_Append( nbr_list, nindx )

    def get_nearest_particles(self, double xi, double yi, double zi,
                              double radius):
        cdef list nbr_list = []
        self.c_get_nearest_particles(xi, yi, zi, radius, nbr_list)
                        
        return nbr_list

    cdef c_brute_force_neighbors(self, double xi, double yi, double zi,
                                 double radius, list nbrs):

        cdef np.ndarray[np.float64_t, ndim=1] x = self.x
        cdef np.ndarray[np.float64_t, ndim=1] y = self.y
        cdef np.ndarray[np.float64_t, ndim=1] z = self.z

        cdef double xj, yj, zj, dist

        cdef long j

        nbrs[:] = []
        for j in range(self.np):
            xj = x[j]; yj = y[j]; zj = z[j]

            dist = (xi-xj)*(xi-xj) + (yi-yj)*(yi-yj) + (zi-zj)*(zi-zj)

            if ( dist < radius*radius ): nbrs.append(j)

    def brute_force_neighbors(self, double xi, double yi, double zi,
                              double radius):

        cdef list nbrs = []
        self.c_brute_force_neighbors(xi,yi,zi,radius,nbrs)

        return nbrs            

    def print_stats(self):
        cdef long nvoxels = 0

        cdef long np

        cdef VoxelId vid
        cdef Voxel voxel

        for vid in self.voxels:
            nvoxels += 1
            voxel = self.voxels[vid]

            np = voxel.num_particles()
            if np > np_max:
                np_max = np

        print "Number of voxels: %d"%(nvoxels),
        print " Number of particles: %d\t np/voxel (avg): %f  (max): %d "\
              %(self.np, float(self.np/nvoxels), np_max)

    def get_voxels(self):
        return self.voxels
