""" Declarations for the NNPS functions """

cimport numpy as np

cdef class VoxelId:
    cdef public int ix, iy, iz

    cdef set(self, int, int, int)

cdef class Voxel:
    
    cdef VoxelId vid
    cdef list indices    

cdef class NNPS:

    cdef dict voxels 

    cdef np.ndarray x
    cdef np.ndarray y
    cdef np.ndarray z

    cdef long np

    cdef double sx, sy, sz

    cdef double Mx, mx, My, my, Mz, mz

    cdef c_bin_particles(self)

    cdef c_get_adjacent_voxels(self, double xi, double yi, double zi,
                               list voxel_list, int  distance=*)

    cdef c_get_nearest_particles(self, double, double, double, double, list)

    cdef c_brute_force_neighbors(self, double, double, double, double, list)
