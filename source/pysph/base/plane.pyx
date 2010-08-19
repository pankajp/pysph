"""
Class representing a plane in 3D.
"""
# Python imports
import numpy
cimport numpy

# local import
from pysph.base.point cimport Point

include "stdlib.pxd"

# The dtype of the float arrays
DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_t

###############################################################################
# `Plane` class.
###############################################################################
cdef class Plane:
    """
    This class represents a Plane in 3D.
    The equation of the plane is assumed to
    be of the form : Ax + By + Cz + D = 0
    """
    #cdef public Point normal
    #cdef public double distance

    ######################################################################
    # `object` interface
    ######################################################################
    def __init__(self, double A=0.0, double B=1.0, double C=0.0, double D=0.0):
        """
        """
        self.normal = Point()
        self.set(A, B, C, D)
        
    cpdef set(self, double A, double B, double C, double D):
        """
        Set the coefficients of the plane.
        <A, B, C> form the normal of the plane.
        """
        self.normal.x = A
        self.normal.y = B
        self.normal.z = C
        
        # normalize the normal
        cdef double mag = numpy.sqrt(self.normal.norm())
                
        self.normal.x /= mag
        self.normal.y /= mag
        self.normal.z /= mag
        
        self.distance = D

    cpdef numpy.ndarray asarray(self):
        """
        Return a numpy array with the coefficients.
        """
        cdef numpy.ndarray[DTYPE_t, ndim=1] r = numpy.empty(4)
        r[0] = self.normal.x
        r[1] = self.normal.y
        r[2] = self.normal.z
        r[3] = self.distance

        return r

    cpdef double point_distance(self, Point pnt):
        """
        Compute the distance of the given point
        from the plane.
        """
        cdef double d 
        d = self.normal.dot(pnt) + self.distance
        return d
        
    cpdef bint is_parallel(self, Plane plane):
        """
        Returns true if the input plane if parallel
        to this plane.
        """
        return 0

    cpdef bint is_perpendicular(self, Plane plane):
        """
        Returns true if the two plane are perpendicular.
        """
        return 0
    
    cpdef double plane_distance(self, Plane plane):
        """
        Returns the shortest distance between two planes.
        """
        return -1.0
