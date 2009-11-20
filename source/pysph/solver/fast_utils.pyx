"""
Some utitlty functions written in cython.
"""
import numpy

cpdef make_nd_array(arrays=[]):
    """
    Makes a proper array of shape (len(a), len(arrays)), from the passed arrays.
    """
    cdef int na = len(arrays)
    cdef int a_size
    if na == 0:
        return None

    # make sure each array given in input is of same size.
    a_size = len(arrays[0])
    for i from 0 <= i < na:
        if <int>len(arrays[0]) != a_size:
            return None
    
    a = numpy.zeros((a_size, na))
    
    for i from 0 <= i < na:
        _a = arrays[i]
        for j from 0 <= j < a_size:
            a[j][i] = _a[j]

    return a
