"""module to test the timings of carray operations"""

import numpy
cimport numpy

from pysph.base.carray import DoubleArray
from pysph.base.carray cimport DoubleArray

from time import time

# the sizes of array toktest
cdef list Ns = [100,1000,10000,100000]

cpdef dict alloc(ns=Ns):
    """test construction of arrays"""
    cdef double t, t1, t2
    cdef int N, K=1000, i
    cdef dict ret = {}
    empty = numpy.empty
    zeros = numpy.zeros
    
    cdef DoubleArray carr
    cdef numpy.ndarray[ndim=1, dtype=numpy.float64_t] narr
    
    for N in ns:
        t = time()
        for i in range(K):
            carr = DoubleArray(N)
        t = time()-t
        t1 = time()
        for i in range(K):
            narr = empty(N)
        t1 = time()-t1
        t2 = time()
        for i in range(K):
            narr = zeros(N)
        t2 = time()-t2
        ret['carr %d'%N] = t/K
        ret['narre %d'%N] = t1/K
        ret['narr %d'%N] = t2/K
    return ret

cpdef dict loopset(ns=Ns):
    """test setting of value in a loop"""
    cdef double t, t1, t2
    cdef int N, i
    cdef dict ret = {}
    empty = numpy.empty
    zeros = numpy.zeros
    
    cdef DoubleArray carr
    cdef numpy.ndarray[ndim=1, dtype=numpy.float64_t] narr
    
    for N in ns:
        carr = DoubleArray(N)
        t = time()
        for i in range(N):
            carr[i] = 1.0
        t = time()-t
        narr = empty(N)
        t1 = time()
        for i in range(N):
            narr[i] = 1.0
        t1 = time()-t1
        t2 = time()
        for i in range(N):
            carr.data[i] = 1.0
        t2 = time()-t2
        ret['carr loopset %d'%N] = t/N
        ret['carrd loopset %d'%N] = t2/N
        ret['narr loopset %d'%N] = t1/N
    return ret

cpdef dict sliceset(ns=Ns):
    """test setting of value using slice operation"""
    cdef double t, t1, t2
    cdef int N, i
    cdef dict ret = {}
    empty = numpy.empty
    zeros = numpy.zeros
    
    cdef DoubleArray carr
    cdef numpy.ndarray[ndim=1, dtype=numpy.float64_t] narr
    
    for N in ns:
        carr = DoubleArray(N)
        t = time()
        #carr[:] = 1.0
        t = time()-t
        narr = empty(N)
        t1 = time()
        narr[:] = 1.0
        t1 = time()-t1
        ret['carr sliceset %d'%N] = t/N
        ret['narr sliceset %d'%N] = t1/N
    return ret

cpdef dict loopget(ns=Ns):
    """test retrieval of value in a loop"""
    cdef double t, t1, t2, num
    cdef int N, i
    cdef dict ret = {}
    empty = numpy.empty
    zeros = numpy.zeros
    cdef dict d = {}
    
    cdef DoubleArray carr
    cdef numpy.ndarray[ndim=1, dtype=numpy.float64_t] narr
    
    for N in ns:
        carr = DoubleArray(N)
        t = time()
        for i in range(N):
            num = carr[i]
        t = time()-t
        d[num] = num
        narr = zeros(N)
        t1 = time()
        for i in range(N):
            num = narr[i]
        t1 = time()-t1
        d[num] = num
        t2 = time()
        for i in range(N):
            num = carr.data[i]
        t2 = time()-t2
        d[num] = num
        ret['carr loopget %d'%N] = t/N
        ret['carrd loopget %d'%N] = t2/N
        ret['narr loopget %d'%N] = t1/N
    return ret

cpdef dict sliceget(ns=Ns):
    """*** not implemented ***"""
    cdef double t, t1, t2
    cdef dict ret = {}
    return ret


cpdef dict loopsum(ns=Ns):
    """test summation of all values in a loop *** not implemented ***"""
    cdef double t, t1, t2
    cdef DoubleArray carr
    cdef numpy.ndarray[ndim=1, dtype=numpy.float] narr
    ret = {}
    
    for N in ns:
        carr = DoubleArray(N)
        narr = numpy.empty(N)
        t = time()
        for i in range(N):
            pass
        t = time()-t
    return ret


# defines the functions which return benchmark numbers dict
cdef list funcs = [alloc, loopset, loopget]


cpdef bench():
    """returns a list of a dict of array operations timings"""
    cdef list timings = []
    for func in funcs:
        timings.append(func())
    return timings
    
if __name__ == '__main__':
    print bench()