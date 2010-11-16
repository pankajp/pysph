cimport numpy

cdef extern from "math.h":
    double sqrt(double)
    double ceil(double)

cdef extern from 'limits.h':
    cdef int INT_MAX

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

cdef class IntPoint:
    cdef readonly int x
    cdef readonly int y
    cdef readonly int z

    cpdef numpy.ndarray asarray(self)
    cdef bint is_equal(self, IntPoint)
    cdef IntPoint diff(self, IntPoint)
    cdef tuple to_tuple(self)
    cdef IntPoint copy(self)

cdef inline Point Point_new(double x, double y, double z):
    cdef Point p = Point.__new__(Point)
    p.x = x
    p.y = y
    p.z = z
    return p

cdef inline Point Point_sub(Point pa, Point pb):
    return Point_new(pa.x-pb.x, pa.y-pb.y, pa.z-pb.z)

cdef inline Point Point_add(Point pa, Point pb):
    return Point_new(pa.x+pb.x, pa.y+pb.y, pa.z+pb.z)


cdef inline IntPoint IntPoint_new(int x, int y, int z):
    cdef IntPoint p = IntPoint.__new__(IntPoint)
    p.x = x
    p.y = y
    p.z = z
    return p

cdef inline IntPoint IntPoint_sub(IntPoint pa, IntPoint pb):
    return IntPoint_new(pa.x-pb.x, pa.y-pb.y, pa.z-pb.z)

cdef inline IntPoint IntPoint_add(IntPoint pa, IntPoint pb):
    return IntPoint_new(pa.x+pb.x, pa.y+pb.y, pa.z+pb.z)

cdef inline double Point_length(Point p):
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z)

cdef inline double Point_length2(Point p):
    return p.x*p.x + p.y*p.y + p.z*p.z

cdef inline double Point_distance(Point pa, Point pb):
    return sqrt((pa.x-pb.x)*(pa.x-pb.x) +
                (pa.y-pb.y)*(pa.y-pb.y) + 
                (pa.z-pb.z)*(pa.z-pb.z)
                )

cdef inline double Point_distance2(Point pa, Point pb):
    return ((pa.x-pb.x)*(pa.x-pb.x) + (pa.y-pb.y)*(pa.y-pb.y) + 
                    (pa.z-pb.z)*(pa.z-pb.z))
