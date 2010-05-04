
"""
Nearest neighbor particle search code.
"""
# Author: Prabhu Ramachandran <prabhu@aero.iitb.ac.in>
# Copyright (c) 2009, Prabhu Ramachandran

# Cython cimports
cimport numpy

# Python imports.
import numpy

# The float dtype to optimize cython.
DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_t


cdef extern from "math.h":
    double fmax(double, double)
    double fmin(double, double)

cdef inline int int_max(int x, int y):
    if x > y:
        return x
    return y

cdef inline int int_min(int x, int y):
    if x < y:
        return x
    return y

cdef inline double square(double dx, double dy, double dz):
    return dx*dx + dy*dy + dz*dz


cpdef brute_force_nnps(Point pnt, double radius, 
                       numpy.ndarray xar, numpy.ndarray yar,
                       numpy.ndarray zar, long exclude_index=-1):
    """Brute force computation of nearest neighbors using an N^2
    algorithm."""
    cdef numpy.ndarray[DTYPE_t, ndim=1] xa = numpy.asarray(xar, dtype=float)
    cdef numpy.ndarray[DTYPE_t, ndim=1] ya = numpy.asarray(yar, dtype=float)
    cdef numpy.ndarray[DTYPE_t, ndim=1] za = numpy.asarray(zar, dtype=float)
    cdef long n = len(xar)
    cdef list ind = []
    cdef list d = []
    cdef double r2 = radius*radius
    cdef long i
    cdef double dist
    for i in range(n):
        dist = square(pnt.x - xa[i], pnt.y - ya[i], pnt.z - za[i])
        if dist <= r2 and i != exclude_index:
            ind.append(i)
            d.append(dist)
    return ind, d

cpdef brute_force_nnps_pm(Point pnt, double radius, 
                          ParticleArray pa, str xn, str yn, str zn,
                          long exclude_index=-1):
    """Brute force computation of nearest neighbors using an N^2
    algorithm.  The function signature is slightly different here.  This
    function is slower than the one using pure numpy arrays."""
    cdef long n = pa.get_number_of_particles()
    cdef list ind = []
    cdef list d = []
    cdef double r2 = radius*radius
    cdef long i
    cdef double dist
    cdef numpy.ndarray x = pa.get(xn)
    cdef numpy.ndarray y = pa.get(yn)
    cdef numpy.ndarray z = pa.get(zn)

    for i in range(n):
        dist = square(pnt.x - x[i], pnt.y - y[i],
                      pnt.z - z[i])
        if dist <= r2 and i != exclude_index:
            ind.append(i)
            d.append(dist)
    return ind, d

###############################################################################
# `NNPS` class.
############################################################################### 
cdef class NNPS:
    """
    This class defines a nearest neighbor particle search algorithm in
    3D for a particle array.
    """

    # Declared in the nnps.pxd file.
    #cdef ParticleArray _pa
    #cdef str _xn, _yn, _zn
    #cdef double _xmin, _ymin, _zmin, _h
    #cdef int _ximax, _yimax, _zimax
    #cdef dict _bin

    ######################################################################
    # `object` interface.
    ###################################################################### 
    def __init__(self, ParticleArray pa, str x='x', str y='y', str z='z'):
        """
        Constructor.  Note that this is built for 3D and in the 1D case
        you must provide additional arrays to perform the indexing.

        Parameters
        ----------
        
        pa -- The particle array.

        x -- The name of the array storing x coordinates.

        y -- The name of the array storing y coordinates.

        z -- The name of the array storing z coordinates.
        """
        self._pa = pa
        self._xn = x
        self._yn = y
        self._zn = z
        self._xmin, self._ymin, self._zmin = 0,0,0
        self._ximax, self._yimax, self._zimax = -1, -1, -1
        self._bin = {}

    ######################################################################
    # `NNPS` interface.
    ###################################################################### 
    cpdef update(self, double bin_size):
        """Calculate the internal data from the particle positions.
        This is meant to be called when the particles change for
        example.
        """
        self._h = bin_size
        cdef double h = bin_size

        # Find the minimum position of the particles.
        cdef double xmin, ymin, zmin
        xmin = ymin = zmin = 1e20
        cdef long i
        cdef long np = self._pa.get_number_of_particles()

        cdef numpy.ndarray x = self._pa.get(self._xn)
        cdef numpy.ndarray y = self._pa.get(self._yn)
        cdef numpy.ndarray z = self._pa.get(self._zn)

        for i in range(np):
            xmin = fmin(xmin, x[i])
            ymin = fmin(ymin, y[i])
            zmin = fmin(zmin, z[i])

        # Store the values found.
        self._xmin, self._ymin, self._zmin = xmin, ymin, zmin

        cdef dict bin = {}
        cdef int xim, xi, yim, yi, zim, zi
        xim = yim = zim = -1
        cdef tuple key

        for i in range(np):
            xi = <int>((x[i] - xmin)/h)
            yi = <int>((y[i] - ymin)/h)
            zi = <int>((z[i] - zmin)/h)
            xim = int_max(xi, xim)
            yim = int_max(yi, yim)
            zim = int_max(zi, zim)
            key = (xi, yi, zi)
            if key in bin:
                bin[key].append(i)
            else:
                bin[key] = [i]
        
        self._bin = bin
        self._ximax, self._yimax, self._zimax = xim, yim, zim

    cpdef tuple get_nearest_particles(self, Point pnt, double radius, 
                              long exclude_index=-1):
        """Given a position as x, y, z, return a list of indices and a
        list of the square of the distance to those points.  It excludes
        any particle with the index specified as the `exclude_index`
        argument.
        """
        cdef double h = self._h
        cdef double xmin, ymin, zmin
        cdef int xim, xi, yim, yi, zim, zi, di
        xmin, ymin, zmin = self._xmin, self._ymin, self._zmin
        xim, yim, zim = self._ximax, self._yimax, self._zimax

        xi = <int>((pnt.x - xmin)/h)
        yi = <int>((pnt.y - ymin)/h)
        zi = <int>((pnt.z - zmin)/h)
        di = <int>(radius/h + 0.5)

        cdef dict bin = self._bin
        cdef int imin, imax, jmin, jmax, kmin, kmax
        imin = int_min(int_max(xi-di, 0), xim) - 1 
        imax = int_max(0, int_min(xi+di, xim)) + 1
        jmin = int_min(int_max(yi-di, 0), yim)
        jmax = int_max(0, int_min(yi+di, yim)) + 1
        kmin = int_min(int_max(zi-di, 0), zim)
        kmax = int_max(0, int_min(zi+di, zim)) + 1
        cdef list indices = []
        cdef list dists = []
        cdef double rsqr = radius*radius
        cdef int i, j, k, m, p_idx, np
        cdef list particles
        cdef double d2
        cdef tuple t
        
        cdef numpy.ndarray x = self._pa.get(self._xn)
        cdef numpy.ndarray y = self._pa.get(self._yn)
        cdef numpy.ndarray z = self._pa.get(self._zn)

        for i in range(imin, imax + 1):
            for j in range(jmin, jmax):
                for k in range(kmin, kmax):
                    particles = bin.get((i, j, k))
                    if particles is not None:
                        np = len(particles)
                        for m in range(np):
                            p_idx = particles[m]
                            if p_idx == exclude_index:
                                continue
                            d2 = square(x[p_idx] - pnt.x, 
                                        y[p_idx] - pnt.y, 
                                        z[p_idx] - pnt.z)
                            if d2 <= rsqr:
                                indices.append(p_idx)
                                dists.append(d2)

        return indices, dists
############################################################################
