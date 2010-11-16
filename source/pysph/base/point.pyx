"""
"""

from cpython cimport *
# numpy imports
cimport numpy
import numpy

# IntPoint's maximum absolute value must be less than `IntPoint_maxint`
# this is due to the hash implementation
cdef int IntPoint_maxint = 2**20

DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_t


###############################################################################
# `Point` class.
############################################################################### 
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

    def __reduce__(self):
        """
        Implemented to facilitate pickling of the Point extension type.
        """
        d = {}
        d['x'] = self.x
        d['y'] = self.y
        d['z'] = self.z

        return (Point, (), d)

    def __setstate__(self, d):
        self.x = d['x']
        self.y = d['y']
        self.z = d['z']

    def __str__(self):
        return '(%f, %f, %f)'%(self.x, self.y, self.z)

    def __repr__(self):
        return 'Point(%g, %g, %g)'%(self.x, self.y, self.z)

    def __add__(self, Point p):
        return Point_new(self.x + p.x, self.y + p.y, self.z + p.z)

    def __sub__(self, Point p):
        return Point_new(self.x - p.x, self.y - p.y, self.z - p.z)

    def __mul__(self, double m):
        return Point_new(self.x*m, self.y*m, self.z*m)

    def __div__(self, double m):
        return Point_new(self.x/m, self.y/m, self.z/m)

    def __abs__(self):
        return self.length()

    def __neg__(self):
        return Point_new(-self.x, -self.y, -self.z)

    def __richcmp__(Point self, Point p, int oper):
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

    cpdef inline double norm(self):
        """Return the square of the Euclidean distance to this point."""
        return (self.x*self.x + self.y*self.y + self.z*self.z)

    cpdef double length(self):
        """Return the Euclidean distance to this point."""
        return sqrt(self.norm())

    cpdef double dot(self, Point p):
        """Return the dot product of this point with another."""
        return self.x*p.x + self.y*p.y + self.z*p.z

    cpdef Point cross(self, Point p):
        """Return the cross product of this point with another, i.e.
        `self` cross `p`."""
        return Point_new(self.y*p.z - self.z*p.y, 
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

    def __reduce__(self):
        """
        Implemented to facilitate pickling of the IntPoint extension type.
        """
        d = {}
        d['x'] = self.x
        d['y'] = self.y
        d['z'] = self.z

        return (IntPoint, (), d)

    def __setstate__(self, d):
        self.x = d['x']
        self.y = d['y']
        self.z = d['z']

    def __str__(self):
        return '(%d,%d,%d)'%(self.x, self.y, self.z)

    def __repr__(self):
        return 'IntPoint(%d,%d,%d)'%(self.x, self.y, self.z)

    cdef IntPoint copy(self):
        return IntPoint_new(self.x, self.y, self.z)

    cpdef numpy.ndarray asarray(self):
        cdef numpy.ndarray[ndim=1,dtype=numpy.int_t] arr = numpy.empty(3,
                                                            dtype=numpy.int)
        arr[0] = self.x
        arr[1] = self.y
        arr[2] = self.z

        return arr

    cdef bint is_equal(self, IntPoint p):
        return (self.x == p.x and self.y == p.y and self.z == p.z)

    cdef IntPoint diff(self, IntPoint p):
        return IntPoint_new(self.x-p.x, self.y-p.y, self.z-p.z)

    cdef tuple to_tuple(self):
        cdef tuple t = (self.x, self.y, self.z)
        return t        

    def __richcmp__(IntPoint self, IntPoint p, int oper):
        # strange cython performance bug (cython 0.13) self is untyped
        if oper == 2: # ==
            return (self.x == p.x and self.y == p.y and self.z == p.z)
        elif oper == 3: # !=
            return not (self.x == p.x and self.y == p.y and self.z == p.z)
        else:
            raise TypeError('No ordering is possible for Points.')

    def __hash__(self):
        cdef long ret = self.x + IntPoint_maxint
        ret = 2 * IntPoint_maxint * ret + self.y + IntPoint_maxint
        return 2 * IntPoint_maxint * ret + self.z + IntPoint_maxint

    def py_is_equal(self, IntPoint p):
        return self.is_equal(p)
    
    def py_diff(self, IntPoint p):
        return self.diff(p)
        
    def py_copy(self):
        return self.copy()
