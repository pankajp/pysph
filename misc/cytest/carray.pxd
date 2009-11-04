# This file has been generated automatically on
# Wed Nov  4 15:47:42 2009
# DO NOT modify this file
# To make changes modify the source templates and regenerate
"""
Implementation of arrays of different types in Cython.

Declaration File.

"""

# numpy import
cimport numpy as np

cdef class BaseArray:
    """
    Base class for managed C-arrays.
    """     
    cdef public int length, alloc
    cdef np.ndarray _npy_array
    
    cpdef reserve(self, int size)
    cpdef resize(self, int size)
    cpdef np.ndarray get_npy_array(self)
    cpdef set_data(self, np.ndarray)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, int input_sorted=*)
    cpdef extend(self, np.ndarray in_array)
    cpdef reset(self)

################################################################################
# `IntArray` class.
################################################################################
cdef class IntArray(BaseArray):
    """
    This class defines a managed array of ints.
    """     
    cdef int *data
        
    cdef _setup_npy_array(self)
    cdef int* get_data_ptr(self)
    
    cpdef int get(self, int idx)
    cpdef set(self, int idx, int value)
    cpdef append(self, int value)
    cpdef reserve(self, int size)
    cpdef resize(self, int size)
    cpdef np.ndarray get_npy_array(self)
    cpdef set_data(self, np.ndarray)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, int input_sorted=*)
    cpdef extend(self, np.ndarray in_array)
    cpdef reset(self)

################################################################################
# `DoubleArray` class.
################################################################################
cdef class DoubleArray(BaseArray):
    """
    This class defines a managed array of doubles.
    """     
    cdef double *data
        
    cdef _setup_npy_array(self)
    cdef double* get_data_ptr(self)
    
    cpdef double get(self, int idx)
    cpdef set(self, int idx, double value)
    cpdef append(self, double value)
    cpdef reserve(self, int size)
    cpdef resize(self, int size)
    cpdef np.ndarray get_npy_array(self)
    cpdef set_data(self, np.ndarray)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, int input_sorted=*)
    cpdef extend(self, np.ndarray in_array)
    cpdef reset(self)

################################################################################
# `FloatArray` class.
################################################################################
cdef class FloatArray(BaseArray):
    """
    This class defines a managed array of floats.
    """     
    cdef float *data
        
    cdef _setup_npy_array(self)
    cdef float* get_data_ptr(self)
    
    cpdef float get(self, int idx)
    cpdef set(self, int idx, float value)
    cpdef append(self, float value)
    cpdef reserve(self, int size)
    cpdef resize(self, int size)
    cpdef np.ndarray get_npy_array(self)
    cpdef set_data(self, np.ndarray)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, int input_sorted=*)
    cpdef extend(self, np.ndarray in_array)
    cpdef reset(self)

################################################################################
# `LongArray` class.
################################################################################
cdef class LongArray(BaseArray):
    """
    This class defines a managed array of longs.
    """     
    cdef long *data
        
    cdef _setup_npy_array(self)
    cdef long* get_data_ptr(self)
    
    cpdef long get(self, int idx)
    cpdef set(self, int idx, long value)
    cpdef append(self, long value)
    cpdef reserve(self, int size)
    cpdef resize(self, int size)
    cpdef np.ndarray get_npy_array(self)
    cpdef set_data(self, np.ndarray)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, int input_sorted=*)
    cpdef extend(self, np.ndarray in_array)
    cpdef reset(self)

