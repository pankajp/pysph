"""module to test the timings of point operations"""

from pysph.base.point import Point, Point_new, Point_add, Point_sub
from pysph.base.point cimport Point, Point_new, Point_add, Point_sub

from pysph.base.point import IntPoint, IntPoint_new, IntPoint_add, IntPoint_sub
from pysph.base.point cimport IntPoint, IntPoint_new, IntPoint_add, IntPoint_sub

from time import time

cdef long N = 1000000

cpdef dict init():
    """point construction and initialization bench"""
    cdef double t = time()
    cdef Point p
    cdef long i
    
    for i in range(N):
        p = Point()
        p.x = 0
        p.y = 0
        p.z = 0
    cdef double t1 = time()-t
    assert p == Point()
    
    t = time()
    for i in range(N):
        p = Point(0,0,0)
    cdef double t2 = time()-t
    assert p == Point()
    
    t = time()
    for i in range(N):
        p = Point_new()
    cdef double t3 = time()-t
    assert p == Point()
    
    t = time()
    for i in range(N):
        p = Point_new(0,0,0)
    cdef double t4 = time()-t
    assert p == Point()
    
    t = time()
    for i in range(N):
        p = Point_new()
        p.x = 0
        p.y = 0
        p.z = 0
    cdef double t5 = time()-t
    assert p == Point()
    
    t = time()
    for i in range(N):
        p = Point_new()
        p.set(0,0,0)
    cdef double t6 = time()-t
    assert p == Point()
    
    return {'Point()':t1/N, 'Point(0,0,0)':t2/N,
            'Point_new()':t3/N, 'Point_new(0,0,0)':t4/N,
            'Point_new()+eset':t5/N, 'Point_new()+set(0,0,0)':t6/N}

cpdef dict add():
    """addition of two points bench"""
    cdef double t = time()
    cdef Point p, p1, p2
    cdef long i
    p1 = Point(1,2,3)
    p2 = Point(0,1,2)
    
    for i in range(N):
        p = p1 + p2
    cdef double t1 = time()-t
    
    assert p.x == 1
    assert p.y == 3
    assert p.z == 5
    
    t = time()
    for i in range(N):
        p = Point_add(p1,p2)
    cdef double t2 = time()-t
    
    assert p.x == 1
    assert p.y == 3
    assert p.z == 5
    
    return {'p1+p2':t1/N,
            'Point_add(p1,p2)':t2/N
            }

cpdef dict sub():
    """subtraction of two points bench"""
    cdef double t = time()
    cdef Point p, p1, p2
    cdef long i
    p1 = Point(1,2,3)
    p2 = Point(0,1,2)
    
    for i in range(N):
        p = p1 - p2
    cdef double t1 = time()-t
    
    assert p.x == 1
    assert p.y == 1
    assert p.z == 1
    
    t = time()
    for i in range(N):
        p = Point_sub(p1,p2)
    cdef double t2 = time()-t
    
    assert p.x == 1
    assert p.y == 1
    assert p.z == 1
    
    return {'p1-p2':t1/N,
            'Point_sub(p1,p2)':t2/N
            }

cpdef dict i_hash():
    """addition of two points bench"""
    cdef IntPoint p
    cdef long i
    cdef double t, t1, t2, t3
    cdef h
    p = IntPoint(1,2,3)
    
    t = time()
    for i in range(N):
        h = p.hash()
    t1 = time()-t
    
    t = time()
    for i in range(N):
        h = p.hash2()
    t2 = time()-t
    
    t = time()
    for i in range(N):
        h = hash(p)
    t3 = time()-t
    
    return {'ipnt hash':t1/N,
            'ipnt hash2':t2/N,
            'ipnt hash()':t3/N
            }


# all benchmark functions defined
cdef list funcs = [init, add, sub, i_hash]


cpdef bench():
    """returns a list of a dict of point operations timings"""
    cdef list timings = []
    for func in funcs:
        timings.append(func())
    return timings

if __name__ == '__main__':
    print bench()
