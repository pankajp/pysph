"""module to test the timings of cell operations"""

from libcpp.vector cimport vector

cimport numpy
import numpy

cimport pysph.base.cell as cell
import pysph.base.cell as cell

from pysph.base.point cimport Point_new, Point_add, Point_sub, IntPoint, cIntPoint

from time import time

# the sizes of array to test, doesn't affect the result
cdef list Ns = [10000]

cpdef dict construct_immediate_neighbor_list(Ns=Ns):
    """time construction of immediate neighbor list"""
    cdef double t, t1, t2
    
    cdef IntPoint cell_id = IntPoint(9,10,11)
    cdef vector[cIntPoint] neighbor_list
    cdef bint include_self = True
    ret = {}
    
    for include_self in (1, 0):
        s = '' if include_self else 's'
        for N in Ns:
            t = time()
            for i in range(N):
                neighbor_list = cell.construct_immediate_neighbor_list(
						cell_id.data, include_self)
            t = time()-t
            assert neighbor_list.size() == (26 + include_self)
            ret['cell immediate_nbr_list %d%s' %(N,s)] = t/N
    
    return ret


# defines the functions which return benchmark numbers dict
cdef list funcs = [construct_immediate_neighbor_list]


cpdef bench():
    """returns a list of a dict of cell operations timings"""
    cdef list timings = []
    for func in funcs:
        timings.append(func())
    return timings
    
if __name__ == '__main__':
    print bench()
    
