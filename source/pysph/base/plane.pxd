"""
Class representing a plane in 3D.
"""
cimport numpy
from pysph.base.point cimport Point

###############################################################################
# `Plane` class.
###############################################################################
cdef class Plane:
    """
    This class represents a Plane in 3D.
    """
    cdef public Point normal
    cdef public double distance

    cpdef set(self, double A, double B, double C, double D)
    cpdef numpy.ndarray asarray(self)
    cpdef double point_distance(self, Point pnt)
    cpdef double plane_distance(self, Plane pln)
    cpdef bint is_parallel(self, Plane pln)
    cpdef bint is_perpendicular(self, Plane pln)
    
