"""module to test the timings of operations on python and c++ stl containers"""

cimport numpy

from pysph.base.point cimport IntPoint, IntPoint_new

import time

from libcpp.vector cimport vector
from libcpp.map cimport map

cdef list Ns = [100, 1e4, 1e6]


cdef extern from 'cPoint.h':
    cdef struct cIntPoint:
        int x, y, z
    long hash(cIntPoint)
    bint operator<(cIntPoint, cIntPoint)

cdef cIntPoint cIntPoint_new(int x=0, int y=0, int z=0):
    cdef cIntPoint ret
    ret.x = x; ret.y = y; ret.z = z
    return ret


cpdef dict set(Ns=Ns):
    """ set time of dict and stl-map"""
    cdef double t, t1, t2
    cdef int i, N
    cdef dict ret = {}
    get_time = time.time
    cdef list tmp = [None]
    
    cdef IntPoint p
    cdef cIntPoint cp
    
    cdef dict d = {}
    cdef map[cIntPoint, cIntPoint] m
    
    for N in Ns:
        pts = []
        t = get_time()
        for i in range(N):
            p = IntPoint_new(i,i+1,i+2)
            d[p] = p
        t = get_time() - t
        tmp[0] = d[p]
        ret['dict set /%d'%N] = t/N
    for N in Ns:
        pts = []
        t = get_time()
        for i in range(N):
            cp = cIntPoint_new(i,i+1,i+2)
            m[cp] = cp
        t = get_time() - t
        tmp[0] = m[cp]
        ret['stl-map set /%d'%N] = t/N
        
    return ret


cpdef dict get(Ns=Ns):
    """ get time of dict and stl-map"""
    cdef double t, t1, t2
    cdef int i, N
    cdef dict ret = {}
    get_time = time.time
    cdef list tmp = [None]
    
    cdef IntPoint p
    cdef cIntPoint cp
    
    cdef dict d = {}
    cdef map[cIntPoint, cIntPoint] m

    for N in Ns:
        pts = []
        t = get_time()
        for i in range(N):
            p = IntPoint_new(i,i+1,i+2)
            d[p] = p
        t = get_time() - t
        tmp[0] = d[p]
        #ret['dict set /%d'%N] = t/N
    for N in Ns:
        pts = []
        t = get_time()
        for i in range(N):
            cp = cIntPoint_new(i,i+1,i+2)
            m[cp] = cp
        t = get_time() - t
        tmp[0] = m[cp]
        #ret['stl-map set /%d'%N] = t/N

    
    for N in Ns:
        pts = []
        t = get_time()
        for i in range(N):
            p = d[p]
        t = get_time() - t
        tmp[0] = p
        ret['dict get /%d'%N] = t/N
    for N in Ns:
        pts = []
        t = get_time()
        for i in range(N):
            cp = m[cp]
        t = get_time() - t
        tmp[0] = cp
        ret['stl-map get /%d'%N] = t/N
    
    return ret


cpdef dict set_get(Ns=Ns):
    """ set+get time of dict and stl-map"""
    cdef double t, t1, t2
    cdef int i, N
    cdef dict ret = {}
    get_time = time.time
    cdef list tmp = [None]
    
    cdef IntPoint p
    cdef cIntPoint cp
    
    cdef dict d = {}
    cdef map[cIntPoint, cIntPoint] m

    for N in Ns:
        pts = []
        t = get_time()
        for i in range(N):
            p = IntPoint_new(i,i+1,i+2)
            d[p] = p
        t = get_time() - t
        tmp[0] = d[p]
        #ret['dict set /%d'%N] = t/N
    for N in Ns:
        pts = []
        t = get_time()
        for i in range(N):
            cp = cIntPoint_new(i,i+1,i+2)
            m[cp] = cp
        t = get_time() - t
        tmp[0] = m[cp]
        #ret['stl-map set /%d'%N] = t/N

    
    for N in Ns:
        pts = []
        t = get_time()
        for i in range(N):
            d[p] = d[p]
        t = get_time() - t
        tmp[0] = p
        ret['dict set-get /%d'%N] = t/N
    for N in Ns:
        pts = []
        t = get_time()
        for i in range(N):
            m[cp] = m[cp]
        t = get_time() - t
        tmp[0] = cp
        ret['stl-map set-get /%d'%N] = t/N
    
    return ret


cpdef dict intpoint(Ns=Ns):
    """ construction time of cIntPoint and IntPoint """
    cdef double t, t2
    cdef int i, N
    cdef dict ret = {}
    get_time = time.time
    cdef list tmp = [None]
    
    cdef IntPoint p
    cdef cIntPoint cp
    
    for N in Ns:
        pts = []
        t = get_time()
        for i in range(N):
            p = IntPoint_new(i,i+1,i+2)
        t = get_time() - t
        tmp[0] = p
        ret['IntPoint() /%d'%N] = t/N
    for N in Ns:
        pts = []
        t = get_time()
        for i in range(N):
            # TODO: how to avoid compiler optimizations here???
            cp = cIntPoint_new(i,i+1,i+2)
        t = get_time() - t
        tmp[0] = cp
        ret['cIntPoint /%d'%N] = t/N
    
    return ret


cpdef dict list_vector(Ns=Ns):
    """ python list vs c++ vector """
    cdef double t, t2
    cdef int i, N
    cdef dict ret = {}
    get_time = time.time
    cdef list l = []
    cdef vector[IntPoint] v
    cdef vector[cIntPoint] v2
    tmp = [None]
    
    cdef IntPoint p = IntPoint_new(1,2,3)
    cdef cIntPoint cp = cIntPoint_new(1,2,3)
    
    for N in Ns:
        pts = []
        t = get_time()
        for i in range(N):
            l.append(p)
        t = get_time() - t
        tmp[0] = l
        ret['list append /%d'%N] = t/N
    for N in Ns:
        pts = []
        t = get_time()
        for i in range(N):
            v.push_back(p)
        t = get_time() - t
        tmp[0] = v[0]
        ret['vector push_back /%d'%N] = t/N
    for N in Ns:
        pts = []
        t = get_time()
        for i in range(N):
            v2.push_back(cp)
        t = get_time() - t
        tmp[0] = v[0]
        ret['vector push_back c /%d'%N] = t/N
    
    return ret

# defines the functions which return benchmark numbers dict
cdef list funcs = [get, set, set_get, intpoint, list_vector]


cpdef bench():
    """returns a list of a dict of cython and c/c++ operation timings"""
    cdef list timings = []
    for func in funcs:
        timings.append(func())
    return timings
    
if __name__ == '__main__':
    print bench()
    
