
import numpy

from pysph.base.carray cimport LongArray, DoubleArray
from pysph.base.point import IntPoint
from pysph.base.point cimport IntPoint

import time

cdef extern from "time.h":
    ctypedef long clock_t
    clock_t clock()

cdef extern from "stdlib.h":
    int RAND_MAX
    int crand "rand" ()

cdef extern from "math.h":
    double sqrt(double)

Ns = [1000, 100000]#, 10000000, 100000000]

cpdef mul_pow():
    cdef dict ret = {}
    tmp = []
    cdef double t, t1, t2, t3
    get_time = time.time
    cdef long l
    cdef double d
    cdef LongArray la = LongArray()
    cdef DoubleArray da = DoubleArray()
    cdef int i, N, k
    for N in Ns:
        d = l = 0
        a = numpy.arange(i)
        da.resize(N)
        da.set_data(a)
        la.resize(N)
        la.set_data(a)
        
        k = 2
        t = get_time()
        for i in range(N):
            d += da.data[i]*da.data[i]
        t = get_time() - t
        ret['double *%d /%d'%(k,N)] = t/N
        
        t = get_time()
        for i in range(N):
            d += da.data[i]**k
        t = get_time() - t
        ret['double **%d /%d'%(k,N)] = t/N
        
        k = 3
        t = get_time()
        for i in range(N):
            d += da.data[i]*da.data[i]*da.data[i]
        t = get_time() - t
        ret['double *%d /%d'%(k,N)] = t/N
        
        t = get_time()
        for i in range(N):
            d += da.data[i]**k
        t = get_time() - t
        ret['double **%d /%d'%(k,N)] = t/N
        
        
        t = get_time()
        for i in range(N):
            d += sqrt(da.data[i])
        t = get_time() - t
        ret['double sqrt /%d'%(N)] = t/N
        #d = 0.5
        t = get_time()
        for i in range(N):
            d += da.data[i]**0.5
        t = get_time() - t
        ret['double **0.5 /%d'%(N)] = t/N
        
        tmp.append(d)
        
    return ret

cpdef dim_test():
    """ test time for calculating 1/h**dim """
    get_time = time.time
    tmp = []
    cdef dict ret = {}
    cdef double t, h=0.1, d=0.0
    cdef IntPoint p = IntPoint(1,2,3)
    cdef int i, N, dim
    
    for N in Ns:
        for dim in [p.x, p.y, p.z]:
            t = get_time()
            for i in range(N):
                d += (1/h)**dim
            t = get_time() - t
            ret['(1/h)**d%d /%d'%(dim,N)] = t/N
            
            t = get_time()
            for i in range(N):
                if dim == 1:
                    d += 1/h
                elif dim == 2:
                    d += 1/(h*h)
                else:
                    d += 1/(h*h*h)
            t = get_time() - t
            ret['if * d%d /%d'%(dim,N)] = t/N
            
            t = get_time()
            for i in range(N):
                d += h**(-dim)
            t = get_time() - t
            ret['h**-d%d /%d'%(dim,N)] = t/N
            
            tmp.append(d)
        
    return ret

cdef list funcs = [mul_pow, dim_test]


cpdef bench():
    cdef list timings = []
    for func in funcs:
        timings.append(func())
    return timings
    
