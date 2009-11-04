"""
"""

# numpy imports
cimport numpy
import numpy
import numpy

cdef extern from "math.h":
    double sqrt(double)
    double ceil(double)

DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_t

################################################################################
# `Point` class.
################################################################################ 
cdef class Point:
    """
    This class represents a Point in 3D space.
    """

    # Declared in the .pxd file.
    #cdef public double x, y, z
    
    ######################################################################
    # `object` interface.
    ######################################################################
    def __init__(self, double x=0.0, double y=0.0, double z=0.0):
        """Constructor for a Point."""
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return '(%f, %f, %f)'%(self.x, self.y, self.z)

    def __add__(self, Point p):
        return Point(self.x + p.x, self.y + p.y, self.z + p.z)

    def __sub__(self, Point p):
        return Point(self.x - p.x, self.y - p.y, self.z - p.z)

    def __mul__(self, double m):
        return Point(self.x*m, self.y*m, self.z*m)

    def __div__(self, double m):
        return Point(self.x/m, self.y/m, self.z/m)

    def __abs__(self):
        return self.length()

    def __neg__(self):
        return Point(-self.x, -self.y, -self.z)

    def __richcmp__(self, Point p, int oper):
        if oper == 2: # ==
            if self.x == p.x and self.y == p.y and self.z == p.z:
                return True
            return False
        elif oper == 3: # !=
            if self.x == p.x and self.y == p.y and self.z == p.z:
                return False 
            return True
        else:
            raise TypeError('No ordering is possible for Points.')

    def __iadd__(self, Point p):
        self.x += p.x
        self.y += p.y
        self.z += p.z
        return self

    def __isub__(self, Point p):
        self.x -= p.x
        self.y -= p.y
        self.z -= p.z
        return self

    def __imul__(self, double m):
        self.x *= m
        self.y *= m
        self.z *= m
        return self

    def __idiv__(self, double m):
        self.x /= m
        self.y /= m
        self.z /= m
        return self

    ######################################################################
    # `Point` interface.
    ######################################################################
    cpdef set(self, double x, double y, double z):
        """Set the position from a given array.
        """
        self.x = x
        self.y = y
        self.z = z

    cpdef numpy.ndarray asarray(self):
        """Return a numpy array with the coordinates."""
        cdef numpy.ndarray[DTYPE_t, ndim=1] r = numpy.empty(3)
        r[0] = self.x
        r[1] = self.y
        r[2] = self.z
        return r

    cpdef double norm(self):
        """Return the square of the Euclidean distance to this point."""
        cdef double x = self.x
        cdef double y = self.y
        cdef double z = self.z
        return (x*x + y*y + z*z)

    cpdef double length(self):
        """Return the Euclidean distance to this point."""
        return sqrt(self.norm())

    cpdef double dot(self, Point p):
        """Return the dot product of this point with another."""
        return self.x*p.x + self.y*p.y + self.z*p.z

    cpdef Point cross(self, Point p):
        """Return the cross product of this point with another, i.e.
        `self` cross `p`."""
        return Point(self.y*p.z - self.z*p.y, 
                     self.z*p.x - self.x*p.z,
                     self.x*p.y - self.y*p.x)

    cpdef double distance(self, Point p):
        """Return the distance between this point and p"""
        return sqrt((p.x-self.x)*(p.x-self.x)+
                    (p.y-self.y)*(p.y-self.y)+
                    (p.z-self.z)*(p.z-self.z))

cdef class IntPoint:
    
    def __init__(self, int x=0, int y=0, int z=0):
        self.x = x
        self.y = y
        self.z = z

    cpdef set(self, int x, int y, int z):
        self.x = x
        self.y = y
        self.z = z

    cdef IntPoint copy(self):
        cdef IntPoint pt = IntPoint()

        pt.x = self.x
        pt.y = self.y
        pt.z = self.z

        return pt

    cpdef numpy.ndarray asarray(self):

        cdef numpy.ndarray arr = numpy.empty(3, dtype=numpy.int)

        arr[0] = self.x
        arr[1] = self.y
        arr[2] = self.z

        return arr

    cdef bint is_equal(self, IntPoint p):
        
        if self.x == p.x and self.y == p.y and self.z == p.z:
            return True
        else:
            return False

    cdef IntPoint diff(self, IntPoint p):
        cdef IntPoint ret  = IntPoint()
        ret.x = self.x - p.x
        ret.y = self.y - p.y
        ret.z = self.z - p.z
        
        return ret

    cdef tuple to_tuple(self):
        """
        """
        cdef tuple t = (self.x, self.y, self.z)
        return t        

    def __richcmp__(self, IntPoint p, int oper):
        if oper == 2: # ==
            if self.x == p.x and self.y == p.y and self.z == p.z:
                return True
            return False
        elif oper == 3: # !=
            if self.x == p.x and self.y == p.y and self.z == p.z:
                return False 
            return True
        else:
            raise TypeError('No ordering is possible for Points.')

    def __hash__(self):
        return (self.x, self.y, self.z).__hash__()

    def __str__(self):
        return '(%d, %d, %d)'%(self.x, self.y, self.z)

    def py_is_equal(self, IntPoint p):
        return self.is_equal(p)
    def py_diff(self, IntPoint p):
        return self.diff(p)
        
    def py_copy(self):
        return self.copy()