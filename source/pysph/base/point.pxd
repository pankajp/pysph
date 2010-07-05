cimport numpy

cdef class Point:
    """
    Class to represent point in 3D.
    """
    cdef public double x
    cdef public double y
    cdef public double z
    
    cpdef set(self, double x, double y, double z)
    cpdef numpy.ndarray asarray(self)
    cpdef double norm(self)
    cpdef double length(self)
    cpdef double dot(self, Point p)
    cpdef Point cross(self, Point p)
    cpdef double distance(self, Point p)

cpdef Point Point_new(double x=*, double y=*, double z=*)
cpdef Point Point_sub(Point pa, Point pb)
cpdef Point Point_add(Point pa, Point pb)


cdef class IntPoint:
    cdef public int x
    cdef public int y
    cdef public int z

    cpdef set(self, int x, int y, int z)
    cpdef numpy.ndarray asarray(self)
    cdef bint is_equal(self, IntPoint)
    cdef IntPoint diff(self, IntPoint)
    cdef tuple to_tuple(self)
    cdef IntPoint copy(self)
