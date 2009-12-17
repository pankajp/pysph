# This file has been generated automatically on
# Wed Dec 16 23:59:27 2009
# DO NOT modify this file
# To make changes modify the source templates and regenerate
"""
Implementation of resizeable arrays of different types in Cython.

All arrays provide for the following operations:

 - access by indexing.
 - access through get/set function.
 - appending values at the end of the array.
 - reserving space for future appends.
 - access to internal data through a numpy array.

** Numpy array access **
Each array also provides an interface to its data through a numpy array. This is
done through the get_npy_array function. The returned numpy array can be used
just like any other numpy array but for the following restrictions:

 - the array may not be resized.
 - references of this array should not be kept.
 - slices of this array may not be made.

The numpy array may however be copied and used in any manner.

** Examples **

"""
# For malloc etc.
include "stdlib.pxd"

cimport numpy as np

import numpy as np

# logging imports
import logging
logger = logging.getLogger()

# 'importing' some Numpy C-api functions.
cdef extern from "numpy/arrayobject.h":
    cdef void  import_array()
    
    ctypedef struct PyArrayObject:
        char  *data
        int *dimensions
    
    cdef enum NPY_TYPES:
        NPY_INT, 
        NPY_LONG,
        NPY_FLOAT,
        NPY_DOUBLE
    
    np.ndarray PyArray_SimpleNewFromData(int, int*, int, void*)
    

# memcpy
cdef extern from "stdlib.h":
     void *memcpy(void *dst, void *src, long n)

# numpy module initialization call
import_array()

# forward declaration
cdef class BaseArray
cdef class LongArray(BaseArray)

cdef class BaseArray:
    """
    Base class for managed C-arrays.
    """     
    def __cinit__(self, *args, **kwargs):
        pass

    cpdef str get_c_type(self):
        """
	Return the c data type of this array.
        """
        raise NotImplementedError, 'BaseArray::get_c_type'

    cpdef reserve(self, int size):
        raise NotImplementedError, 'BaseArray::reserve'

    cpdef resize(self, int size):
        raise NotImplementedError, 'BaseArray::resize'

    cpdef np.ndarray get_npy_array(self):
        return self._npy_array

    cpdef set_data(self, np.ndarray nparr):
        """
        Set data from the given numpy array.

        If the numpy array is a reference to the numpy array maintained
        internally by this class, nothing is done. 
        Otherwise, if the size of nparr matches this array, values are
        copied into the array maintained.

        """
        cdef PyArrayObject* sarr = <PyArrayObject*>nparr
        cdef PyArrayObject* darr = <PyArrayObject*>self._npy_array

        if sarr.data == darr.data:
            return
        elif sarr.dimensions[0] <= darr.dimensions[0]:
            self._npy_array[:sarr.dimensions[0]] = nparr
        else:
            raise ValueError, 'array size mismatch'

    cpdef squeeze(self):
        raise NotImplementedError, 'BaseArray::squeeze'

    cpdef remove(self, np.ndarray index_list, int input_sorted=0):
        raise NotImplementedError, 'BaseArray::remove'

    cpdef extend(self, np.ndarray in_array):
        raise NotImplementedError, 'BaseArray::extend'

    cpdef align_array(self, LongArray new_indices):
        self._align_array(new_indices)

    cdef void _align_array(self, LongArray new_indices):
        raise NotImplementedError, 'BaseArray::_align_array'	
        
    cpdef reset(self):
        """
        Reset the length of the array to 0.
    	"""
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        self.length = 0
        arr.dimensions[0] = self.length        
        
    cpdef copy_values(self, LongArray indices, BaseArray dest):
        """
        Copy values of selected particles in indices from self to dest.
        """
        raise NotImplementedError, 'BaseArray::copy_values'

    cpdef copy_subset(self, BaseArray source, long start_index=-1, long end_index=-1):
        """
	Copy subset of values from source to self.
	"""
        raise NotImplementedError, 'BaseArray::copy_subset'

    cpdef update_min_max(self):
        """
	Update the min and max values of the array.
	"""
        raise NotImplementedError, 'BaseArray::update_min_max'
################################################################################
# `IntArray` class.
################################################################################
cdef class IntArray(BaseArray):
    #cdef public int length, alloc
    #cdef int *data
    #cdef np.ndarray _npy_array

    def __cinit__(self, int n=0, *args, **kwargs):
        """
        Constructor.
        """
        self.length = n
        if n == 0:
            n = 16
        self.alloc = n
        self.data = <int*>malloc(n*sizeof(int))
        
        self._setup_npy_array()
	 
    def __dealloc__(self):
        """
        Frees the array.
        """
        free(<void*>self.data)
    
    def __getitem__(self, int idx):
        """
        Get item at position idx.
        """
        return self.data[idx]

    def __setitem__(self, int idx, int value):
        """
        Set location idx to value.
        """
        self.data[idx] = value

    cdef _setup_npy_array(self):
        """
        Create the numpy array.
        """
        cdef int nd = 1
        cdef np.npy_intp dims = self.length

        self._npy_array = PyArray_SimpleNewFromData(nd, &dims, NPY_INT, self.data)

    cpdef str get_c_type(self):
        """
        Return the c data type for this array.
        """
        return 'int'

    cdef int* get_data_ptr(self):
        """
        Return the internal data pointer.
        """
        return self.data
            
    cpdef int get(self, int idx):
        """
        Gets value stored at position idx.
        """
        return self.data[idx]

    cpdef set(self, int idx, int value):
        """
        Sets location idx to value.
        """
        self.data[idx] = value
    
    cpdef append(self, int value):
        """
        Appends value to the end of the array.
        """
        cdef int l = self.length
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        if l >= self.alloc:
            self.reserve(l*2)
        self.data[l] = value
        self.length += 1

        # update the numpy arrays length
        arr.dimensions[0] = self.length

    cpdef reserve(self, int size):
        """
        Resizes the internal data to size*sizeof(int) bytes.
        """
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        cdef void* data = NULL
        if size > self.alloc:
            data = <int*>realloc(self.data, size*sizeof(int))

            if data == NULL:
                free(<void*>self.data)
                raise MemoryError

            self.data = <int*>data
            self.alloc = size
            arr.data = <char *>self.data
            
    cpdef resize(self, int size):
        """
 	Resizes internal data to size*sizeof(int) bytes and sets the
        length to the new size.
        
        """
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        # reserve memory
        self.reserve(size)

        # update the lengths
        self.length = size
        arr.dimensions[0] = self.length

    cpdef squeeze(self):
        """
        Release any unused memory.
        """
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        cdef void* data = NULL
        data = <int*>realloc(self.data, self.length*sizeof(int))
        
        if data == NULL:
            # free original data
            free(<void*>self.data)
            raise MemoryError
        
        self.data = <int*>data
        self.alloc = self.length
        arr.data = <char *>self.data
        
    cpdef remove(self, np.ndarray index_list, int input_sorted=0):
        """
        Remove the particles with indices in index_list.

        **Parameters**

         - index_list - a list of indices which should be removed.
         - input_sorted - indicates if the input is sorted in ascending order.
           if not, the array will be sorted internally.

        **Algorithm**
         
         If the input indices are not sorted, sort them in ascending order. 
         Starting with the last element in the index list, start replacing the 
         element at the said index with the last element in the data and update 
         the length of the array.

        """
        cdef int i
        cdef int inlength = index_list.size
        cdef np.ndarray sorted_indices
        cdef long id
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        
        if inlength > self.length:
            return

        if input_sorted != 1:
            sorted_indices = np.sort(index_list)
        else:
            sorted_indices = index_list
        
        for i in range(inlength):
            id = sorted_indices[inlength-(i+1)]
            if id < self.length:
                self.data[id] = self.data[self.length-1]
                self.length = self.length - 1
                arr.dimensions[0] = self.length

    cpdef extend(self, np.ndarray in_array):
        """
        Extend the array with data from in_array.
        
        **Parameters**
         
         - in_array - a numpy array with data to be added to the current array.

        **Issues**
         
         - accessing the in_array using the indexing operation seems to be 
           costly. Look at the annotated cython html file.

        """
        cdef int len = in_array.size
        cdef int i
        for i in range(len):
            self.append(in_array[i])
    
    cdef void _align_array(self, LongArray new_indices):
        """
	Rearrange the contents of the array according to the new indices.
	"""
        if new_indices.length != self.length:
            raise ValueError, 'Unequal array lengths'
	
        cdef int i
        cdef int length = self.length
        cdef int n_bytes
        cdef int *temp
        
        n_bytes = sizeof(int)*length
        temp = <int*>malloc(n_bytes)

        memcpy(<void*>temp, <void*>self.data, n_bytes)

        # copy the data from the resized portion to the actual positions.
        for i from 0 <= i < length:
            if i != new_indices.data[i]:
                self.data[i] = temp[new_indices.data[i]]
        
        free(<void*>temp)

    cpdef copy_values(self, LongArray indices, BaseArray dest):
        """
	Copies values of indices in indices from self to dest.

	No size check if performed, we assume the dest to of proper size
        i.e. atleast as long as indices.
        """
        cdef IntArray dest_array = <IntArray>dest
        cdef int i, num_values
        num_values = indices.length
        
        for i from 0 <= i < num_values:
            dest_array.data[i] = self.data[indices.data[i]]

    cpdef copy_subset(self, BaseArray source, long start_index=-1,
                      long end_index=-1):
        """
        Copy a subset of values from src to self.

        **Parameters**
        
            - start_index - the first index in dest that corresponds to the 0th
            index in source.
            - end_index   - the last index in dest that corresponds to the last
            index in source.

        """
        cdef int si, ei, s_length, d_length, i, j
        cdef IntArray src = <IntArray>source
        s_length = src.length
        d_length = self.length

        if end_index < 0:
            if start_index < 0:
                if s_length != d_length:
                    msg = 'Source length should be same as dest length'
                    logger.error(msg)
                    raise ValueError, msg
                si = 0
                ei = self.length - 1
            else:
                # meaning we copy from the specified start index to the end of
                # self. make sure the sizes are consistent.
                si = start_index
                ei = d_length-1

                if start_index > (d_length-1):
                    msg = 'start_index beyond array length'
                    logger.error(msg)
                    raise ValueError, msg

                if (ei - si + 1) > s_length:
                    msg = 'Not enough values in source'
                    logger.error(msg)
                    raise ValueError, msg
        else:
            if start_index < 0:
                msg = 'start_index : %d, end_index : %d'%(start_index,
                                                          end_index)
                logger.error(msg)
                raise ValueError, msg
            else:
                if (start_index > (d_length-1) or end_index > (d_length-1) or
                    start_index > end_index):
                    msg = 'start_index : %d, end_index : %d'%(start_index,
                                                              end_index)
                    logger.error(msg)
                    raise ValueError, msg

                si = start_index
                ei = end_index

        # we have valid start and end indices now. can start copying now.
        j = 0
        for i from si <= i <= ei:
            self.data[i] = src.data[j]
            j += 1

    cpdef update_min_max(self):
        """
        Updates the min and max values of the array.
        """
        cdef int i = 0
        cdef int min_val, max_val
        
        if self.length == 0:
            self.minimum = <int>-1e20
            self.maximum = <int>1e20
            return

        min_val = self.data[0]
        max_val = self.data[0]

        for i from 0 <= i < self.length:
            if min_val > self.data[i]:
                min_val = self.data[i]
            if max_val < self.data[i]:
                max_val = self.data[i]
        
        self.minimum = min_val
        self.maximum = max_val		   


################################################################################
# `DoubleArray` class.
################################################################################
cdef class DoubleArray(BaseArray):
    #cdef public int length, alloc
    #cdef double *data
    #cdef np.ndarray _npy_array

    def __cinit__(self, int n=0, *args, **kwargs):
        """
        Constructor.
        """
        self.length = n
        if n == 0:
            n = 16
        self.alloc = n
        self.data = <double*>malloc(n*sizeof(double))
        
        self._setup_npy_array()
	 
    def __dealloc__(self):
        """
        Frees the array.
        """
        free(<void*>self.data)
    
    def __getitem__(self, int idx):
        """
        Get item at position idx.
        """
        return self.data[idx]

    def __setitem__(self, int idx, double value):
        """
        Set location idx to value.
        """
        self.data[idx] = value

    cdef _setup_npy_array(self):
        """
        Create the numpy array.
        """
        cdef int nd = 1
        cdef np.npy_intp dims = self.length

        self._npy_array = PyArray_SimpleNewFromData(nd, &dims, NPY_DOUBLE, self.data)

    cpdef str get_c_type(self):
        """
        Return the c data type for this array.
        """
        return 'double'

    cdef double* get_data_ptr(self):
        """
        Return the internal data pointer.
        """
        return self.data
            
    cpdef double get(self, int idx):
        """
        Gets value stored at position idx.
        """
        return self.data[idx]

    cpdef set(self, int idx, double value):
        """
        Sets location idx to value.
        """
        self.data[idx] = value
    
    cpdef append(self, double value):
        """
        Appends value to the end of the array.
        """
        cdef int l = self.length
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        if l >= self.alloc:
            self.reserve(l*2)
        self.data[l] = value
        self.length += 1

        # update the numpy arrays length
        arr.dimensions[0] = self.length

    cpdef reserve(self, int size):
        """
        Resizes the internal data to size*sizeof(double) bytes.
        """
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        cdef void* data = NULL
        if size > self.alloc:
            data = <double*>realloc(self.data, size*sizeof(double))

            if data == NULL:
                free(<void*>self.data)
                raise MemoryError

            self.data = <double*>data
            self.alloc = size
            arr.data = <char *>self.data
            
    cpdef resize(self, int size):
        """
 	Resizes internal data to size*sizeof(double) bytes and sets the
        length to the new size.
        
        """
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        # reserve memory
        self.reserve(size)

        # update the lengths
        self.length = size
        arr.dimensions[0] = self.length

    cpdef squeeze(self):
        """
        Release any unused memory.
        """
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        cdef void* data = NULL
        data = <double*>realloc(self.data, self.length*sizeof(double))
        
        if data == NULL:
            # free original data
            free(<void*>self.data)
            raise MemoryError
        
        self.data = <double*>data
        self.alloc = self.length
        arr.data = <char *>self.data
        
    cpdef remove(self, np.ndarray index_list, int input_sorted=0):
        """
        Remove the particles with indices in index_list.

        **Parameters**

         - index_list - a list of indices which should be removed.
         - input_sorted - indicates if the input is sorted in ascending order.
           if not, the array will be sorted internally.

        **Algorithm**
         
         If the input indices are not sorted, sort them in ascending order. 
         Starting with the last element in the index list, start replacing the 
         element at the said index with the last element in the data and update 
         the length of the array.

        """
        cdef int i
        cdef int inlength = index_list.size
        cdef np.ndarray sorted_indices
        cdef long id
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        
        if inlength > self.length:
            return

        if input_sorted != 1:
            sorted_indices = np.sort(index_list)
        else:
            sorted_indices = index_list
        
        for i in range(inlength):
            id = sorted_indices[inlength-(i+1)]
            if id < self.length:
                self.data[id] = self.data[self.length-1]
                self.length = self.length - 1
                arr.dimensions[0] = self.length

    cpdef extend(self, np.ndarray in_array):
        """
        Extend the array with data from in_array.
        
        **Parameters**
         
         - in_array - a numpy array with data to be added to the current array.

        **Issues**
         
         - accessing the in_array using the indexing operation seems to be 
           costly. Look at the annotated cython html file.

        """
        cdef int len = in_array.size
        cdef int i
        for i in range(len):
            self.append(in_array[i])
    
    cdef void _align_array(self, LongArray new_indices):
        """
	Rearrange the contents of the array according to the new indices.
	"""
        if new_indices.length != self.length:
            raise ValueError, 'Unequal array lengths'
	
        cdef int i
        cdef int length = self.length
        cdef int n_bytes
        cdef double *temp
        
        n_bytes = sizeof(double)*length
        temp = <double*>malloc(n_bytes)

        memcpy(<void*>temp, <void*>self.data, n_bytes)

        # copy the data from the resized portion to the actual positions.
        for i from 0 <= i < length:
            if i != new_indices.data[i]:
                self.data[i] = temp[new_indices.data[i]]
        
        free(<void*>temp)

    cpdef copy_values(self, LongArray indices, BaseArray dest):
        """
	Copies values of indices in indices from self to dest.

	No size check if performed, we assume the dest to of proper size
        i.e. atleast as long as indices.
        """
        cdef DoubleArray dest_array = <DoubleArray>dest
        cdef int i, num_values
        num_values = indices.length
        
        for i from 0 <= i < num_values:
            dest_array.data[i] = self.data[indices.data[i]]

    cpdef copy_subset(self, BaseArray source, long start_index=-1,
                      long end_index=-1):
        """
        Copy a subset of values from src to self.

        **Parameters**
        
            - start_index - the first index in dest that corresponds to the 0th
            index in source.
            - end_index   - the last index in dest that corresponds to the last
            index in source.

        """
        cdef int si, ei, s_length, d_length, i, j
        cdef DoubleArray src = <DoubleArray>source
        s_length = src.length
        d_length = self.length

        if end_index < 0:
            if start_index < 0:
                if s_length != d_length:
                    msg = 'Source length should be same as dest length'
                    logger.error(msg)
                    raise ValueError, msg
                si = 0
                ei = self.length - 1
            else:
                # meaning we copy from the specified start index to the end of
                # self. make sure the sizes are consistent.
                si = start_index
                ei = d_length-1

                if start_index > (d_length-1):
                    msg = 'start_index beyond array length'
                    logger.error(msg)
                    raise ValueError, msg

                if (ei - si + 1) > s_length:
                    msg = 'Not enough values in source'
                    logger.error(msg)
                    raise ValueError, msg
        else:
            if start_index < 0:
                msg = 'start_index : %d, end_index : %d'%(start_index,
                                                          end_index)
                logger.error(msg)
                raise ValueError, msg
            else:
                if (start_index > (d_length-1) or end_index > (d_length-1) or
                    start_index > end_index):
                    msg = 'start_index : %d, end_index : %d'%(start_index,
                                                              end_index)
                    logger.error(msg)
                    raise ValueError, msg

                si = start_index
                ei = end_index

        # we have valid start and end indices now. can start copying now.
        j = 0
        for i from si <= i <= ei:
            self.data[i] = src.data[j]
            j += 1

    cpdef update_min_max(self):
        """
        Updates the min and max values of the array.
        """
        cdef int i = 0
        cdef double min_val, max_val
        
        if self.length == 0:
            self.minimum = <double>-1e20
            self.maximum = <double>1e20
            return

        min_val = self.data[0]
        max_val = self.data[0]

        for i from 0 <= i < self.length:
            if min_val > self.data[i]:
                min_val = self.data[i]
            if max_val < self.data[i]:
                max_val = self.data[i]
        
        self.minimum = min_val
        self.maximum = max_val		   


################################################################################
# `FloatArray` class.
################################################################################
cdef class FloatArray(BaseArray):
    #cdef public int length, alloc
    #cdef float *data
    #cdef np.ndarray _npy_array

    def __cinit__(self, int n=0, *args, **kwargs):
        """
        Constructor.
        """
        self.length = n
        if n == 0:
            n = 16
        self.alloc = n
        self.data = <float*>malloc(n*sizeof(float))
        
        self._setup_npy_array()
	 
    def __dealloc__(self):
        """
        Frees the array.
        """
        free(<void*>self.data)
    
    def __getitem__(self, int idx):
        """
        Get item at position idx.
        """
        return self.data[idx]

    def __setitem__(self, int idx, float value):
        """
        Set location idx to value.
        """
        self.data[idx] = value

    cdef _setup_npy_array(self):
        """
        Create the numpy array.
        """
        cdef int nd = 1
        cdef np.npy_intp dims = self.length

        self._npy_array = PyArray_SimpleNewFromData(nd, &dims, NPY_FLOAT, self.data)

    cpdef str get_c_type(self):
        """
        Return the c data type for this array.
        """
        return 'float'

    cdef float* get_data_ptr(self):
        """
        Return the internal data pointer.
        """
        return self.data
            
    cpdef float get(self, int idx):
        """
        Gets value stored at position idx.
        """
        return self.data[idx]

    cpdef set(self, int idx, float value):
        """
        Sets location idx to value.
        """
        self.data[idx] = value
    
    cpdef append(self, float value):
        """
        Appends value to the end of the array.
        """
        cdef int l = self.length
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        if l >= self.alloc:
            self.reserve(l*2)
        self.data[l] = value
        self.length += 1

        # update the numpy arrays length
        arr.dimensions[0] = self.length

    cpdef reserve(self, int size):
        """
        Resizes the internal data to size*sizeof(float) bytes.
        """
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        cdef void* data = NULL
        if size > self.alloc:
            data = <float*>realloc(self.data, size*sizeof(float))

            if data == NULL:
                free(<void*>self.data)
                raise MemoryError

            self.data = <float*>data
            self.alloc = size
            arr.data = <char *>self.data
            
    cpdef resize(self, int size):
        """
 	Resizes internal data to size*sizeof(float) bytes and sets the
        length to the new size.
        
        """
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        # reserve memory
        self.reserve(size)

        # update the lengths
        self.length = size
        arr.dimensions[0] = self.length

    cpdef squeeze(self):
        """
        Release any unused memory.
        """
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        cdef void* data = NULL
        data = <float*>realloc(self.data, self.length*sizeof(float))
        
        if data == NULL:
            # free original data
            free(<void*>self.data)
            raise MemoryError
        
        self.data = <float*>data
        self.alloc = self.length
        arr.data = <char *>self.data
        
    cpdef remove(self, np.ndarray index_list, int input_sorted=0):
        """
        Remove the particles with indices in index_list.

        **Parameters**

         - index_list - a list of indices which should be removed.
         - input_sorted - indicates if the input is sorted in ascending order.
           if not, the array will be sorted internally.

        **Algorithm**
         
         If the input indices are not sorted, sort them in ascending order. 
         Starting with the last element in the index list, start replacing the 
         element at the said index with the last element in the data and update 
         the length of the array.

        """
        cdef int i
        cdef int inlength = index_list.size
        cdef np.ndarray sorted_indices
        cdef long id
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        
        if inlength > self.length:
            return

        if input_sorted != 1:
            sorted_indices = np.sort(index_list)
        else:
            sorted_indices = index_list
        
        for i in range(inlength):
            id = sorted_indices[inlength-(i+1)]
            if id < self.length:
                self.data[id] = self.data[self.length-1]
                self.length = self.length - 1
                arr.dimensions[0] = self.length

    cpdef extend(self, np.ndarray in_array):
        """
        Extend the array with data from in_array.
        
        **Parameters**
         
         - in_array - a numpy array with data to be added to the current array.

        **Issues**
         
         - accessing the in_array using the indexing operation seems to be 
           costly. Look at the annotated cython html file.

        """
        cdef int len = in_array.size
        cdef int i
        for i in range(len):
            self.append(in_array[i])
    
    cdef void _align_array(self, LongArray new_indices):
        """
	Rearrange the contents of the array according to the new indices.
	"""
        if new_indices.length != self.length:
            raise ValueError, 'Unequal array lengths'
	
        cdef int i
        cdef int length = self.length
        cdef int n_bytes
        cdef float *temp
        
        n_bytes = sizeof(float)*length
        temp = <float*>malloc(n_bytes)

        memcpy(<void*>temp, <void*>self.data, n_bytes)

        # copy the data from the resized portion to the actual positions.
        for i from 0 <= i < length:
            if i != new_indices.data[i]:
                self.data[i] = temp[new_indices.data[i]]
        
        free(<void*>temp)

    cpdef copy_values(self, LongArray indices, BaseArray dest):
        """
	Copies values of indices in indices from self to dest.

	No size check if performed, we assume the dest to of proper size
        i.e. atleast as long as indices.
        """
        cdef FloatArray dest_array = <FloatArray>dest
        cdef int i, num_values
        num_values = indices.length
        
        for i from 0 <= i < num_values:
            dest_array.data[i] = self.data[indices.data[i]]

    cpdef copy_subset(self, BaseArray source, long start_index=-1,
                      long end_index=-1):
        """
        Copy a subset of values from src to self.

        **Parameters**
        
            - start_index - the first index in dest that corresponds to the 0th
            index in source.
            - end_index   - the last index in dest that corresponds to the last
            index in source.

        """
        cdef int si, ei, s_length, d_length, i, j
        cdef FloatArray src = <FloatArray>source
        s_length = src.length
        d_length = self.length

        if end_index < 0:
            if start_index < 0:
                if s_length != d_length:
                    msg = 'Source length should be same as dest length'
                    logger.error(msg)
                    raise ValueError, msg
                si = 0
                ei = self.length - 1
            else:
                # meaning we copy from the specified start index to the end of
                # self. make sure the sizes are consistent.
                si = start_index
                ei = d_length-1

                if start_index > (d_length-1):
                    msg = 'start_index beyond array length'
                    logger.error(msg)
                    raise ValueError, msg

                if (ei - si + 1) > s_length:
                    msg = 'Not enough values in source'
                    logger.error(msg)
                    raise ValueError, msg
        else:
            if start_index < 0:
                msg = 'start_index : %d, end_index : %d'%(start_index,
                                                          end_index)
                logger.error(msg)
                raise ValueError, msg
            else:
                if (start_index > (d_length-1) or end_index > (d_length-1) or
                    start_index > end_index):
                    msg = 'start_index : %d, end_index : %d'%(start_index,
                                                              end_index)
                    logger.error(msg)
                    raise ValueError, msg

                si = start_index
                ei = end_index

        # we have valid start and end indices now. can start copying now.
        j = 0
        for i from si <= i <= ei:
            self.data[i] = src.data[j]
            j += 1

    cpdef update_min_max(self):
        """
        Updates the min and max values of the array.
        """
        cdef int i = 0
        cdef float min_val, max_val
        
        if self.length == 0:
            self.minimum = <float>-1e20
            self.maximum = <float>1e20
            return

        min_val = self.data[0]
        max_val = self.data[0]

        for i from 0 <= i < self.length:
            if min_val > self.data[i]:
                min_val = self.data[i]
            if max_val < self.data[i]:
                max_val = self.data[i]
        
        self.minimum = min_val
        self.maximum = max_val		   


################################################################################
# `LongArray` class.
################################################################################
cdef class LongArray(BaseArray):
    #cdef public int length, alloc
    #cdef long *data
    #cdef np.ndarray _npy_array

    def __cinit__(self, int n=0, *args, **kwargs):
        """
        Constructor.
        """
        self.length = n
        if n == 0:
            n = 16
        self.alloc = n
        self.data = <long*>malloc(n*sizeof(long))
        
        self._setup_npy_array()
	 
    def __dealloc__(self):
        """
        Frees the array.
        """
        free(<void*>self.data)
    
    def __getitem__(self, int idx):
        """
        Get item at position idx.
        """
        return self.data[idx]

    def __setitem__(self, int idx, long value):
        """
        Set location idx to value.
        """
        self.data[idx] = value

    cdef _setup_npy_array(self):
        """
        Create the numpy array.
        """
        cdef int nd = 1
        cdef np.npy_intp dims = self.length

        self._npy_array = PyArray_SimpleNewFromData(nd, &dims, NPY_LONG, self.data)

    cpdef str get_c_type(self):
        """
        Return the c data type for this array.
        """
        return 'long'

    cdef long* get_data_ptr(self):
        """
        Return the internal data pointer.
        """
        return self.data
            
    cpdef long get(self, int idx):
        """
        Gets value stored at position idx.
        """
        return self.data[idx]

    cpdef set(self, int idx, long value):
        """
        Sets location idx to value.
        """
        self.data[idx] = value
    
    cpdef append(self, long value):
        """
        Appends value to the end of the array.
        """
        cdef int l = self.length
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        if l >= self.alloc:
            self.reserve(l*2)
        self.data[l] = value
        self.length += 1

        # update the numpy arrays length
        arr.dimensions[0] = self.length

    cpdef reserve(self, int size):
        """
        Resizes the internal data to size*sizeof(long) bytes.
        """
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        cdef void* data = NULL
        if size > self.alloc:
            data = <long*>realloc(self.data, size*sizeof(long))

            if data == NULL:
                free(<void*>self.data)
                raise MemoryError

            self.data = <long*>data
            self.alloc = size
            arr.data = <char *>self.data
            
    cpdef resize(self, int size):
        """
 	Resizes internal data to size*sizeof(long) bytes and sets the
        length to the new size.
        
        """
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array

        # reserve memory
        self.reserve(size)

        # update the lengths
        self.length = size
        arr.dimensions[0] = self.length

    cpdef squeeze(self):
        """
        Release any unused memory.
        """
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        cdef void* data = NULL
        data = <long*>realloc(self.data, self.length*sizeof(long))
        
        if data == NULL:
            # free original data
            free(<void*>self.data)
            raise MemoryError
        
        self.data = <long*>data
        self.alloc = self.length
        arr.data = <char *>self.data
        
    cpdef remove(self, np.ndarray index_list, int input_sorted=0):
        """
        Remove the particles with indices in index_list.

        **Parameters**

         - index_list - a list of indices which should be removed.
         - input_sorted - indicates if the input is sorted in ascending order.
           if not, the array will be sorted internally.

        **Algorithm**
         
         If the input indices are not sorted, sort them in ascending order. 
         Starting with the last element in the index list, start replacing the 
         element at the said index with the last element in the data and update 
         the length of the array.

        """
        cdef int i
        cdef int inlength = index_list.size
        cdef np.ndarray sorted_indices
        cdef long id
        cdef PyArrayObject* arr = <PyArrayObject*>self._npy_array
        
        if inlength > self.length:
            return

        if input_sorted != 1:
            sorted_indices = np.sort(index_list)
        else:
            sorted_indices = index_list
        
        for i in range(inlength):
            id = sorted_indices[inlength-(i+1)]
            if id < self.length:
                self.data[id] = self.data[self.length-1]
                self.length = self.length - 1
                arr.dimensions[0] = self.length

    cpdef extend(self, np.ndarray in_array):
        """
        Extend the array with data from in_array.
        
        **Parameters**
         
         - in_array - a numpy array with data to be added to the current array.

        **Issues**
         
         - accessing the in_array using the indexing operation seems to be 
           costly. Look at the annotated cython html file.

        """
        cdef int len = in_array.size
        cdef int i
        for i in range(len):
            self.append(in_array[i])
    
    cdef void _align_array(self, LongArray new_indices):
        """
	Rearrange the contents of the array according to the new indices.
	"""
        if new_indices.length != self.length:
            raise ValueError, 'Unequal array lengths'
	
        cdef int i
        cdef int length = self.length
        cdef int n_bytes
        cdef long *temp
        
        n_bytes = sizeof(long)*length
        temp = <long*>malloc(n_bytes)

        memcpy(<void*>temp, <void*>self.data, n_bytes)

        # copy the data from the resized portion to the actual positions.
        for i from 0 <= i < length:
            if i != new_indices.data[i]:
                self.data[i] = temp[new_indices.data[i]]
        
        free(<void*>temp)

    cpdef copy_values(self, LongArray indices, BaseArray dest):
        """
	Copies values of indices in indices from self to dest.

	No size check if performed, we assume the dest to of proper size
        i.e. atleast as long as indices.
        """
        cdef LongArray dest_array = <LongArray>dest
        cdef int i, num_values
        num_values = indices.length
        
        for i from 0 <= i < num_values:
            dest_array.data[i] = self.data[indices.data[i]]

    cpdef copy_subset(self, BaseArray source, long start_index=-1,
                      long end_index=-1):
        """
        Copy a subset of values from src to self.

        **Parameters**
        
            - start_index - the first index in dest that corresponds to the 0th
            index in source.
            - end_index   - the last index in dest that corresponds to the last
            index in source.

        """
        cdef int si, ei, s_length, d_length, i, j
        cdef LongArray src = <LongArray>source
        s_length = src.length
        d_length = self.length

        if end_index < 0:
            if start_index < 0:
                if s_length != d_length:
                    msg = 'Source length should be same as dest length'
                    logger.error(msg)
                    raise ValueError, msg
                si = 0
                ei = self.length - 1
            else:
                # meaning we copy from the specified start index to the end of
                # self. make sure the sizes are consistent.
                si = start_index
                ei = d_length-1

                if start_index > (d_length-1):
                    msg = 'start_index beyond array length'
                    logger.error(msg)
                    raise ValueError, msg

                if (ei - si + 1) > s_length:
                    msg = 'Not enough values in source'
                    logger.error(msg)
                    raise ValueError, msg
        else:
            if start_index < 0:
                msg = 'start_index : %d, end_index : %d'%(start_index,
                                                          end_index)
                logger.error(msg)
                raise ValueError, msg
            else:
                if (start_index > (d_length-1) or end_index > (d_length-1) or
                    start_index > end_index):
                    msg = 'start_index : %d, end_index : %d'%(start_index,
                                                              end_index)
                    logger.error(msg)
                    raise ValueError, msg

                si = start_index
                ei = end_index

        # we have valid start and end indices now. can start copying now.
        j = 0
        for i from si <= i <= ei:
            self.data[i] = src.data[j]
            j += 1

    cpdef update_min_max(self):
        """
        Updates the min and max values of the array.
        """
        cdef int i = 0
        cdef long min_val, max_val
        
        if self.length == 0:
            self.minimum = <long>-1e20
            self.maximum = <long>1e20
            return

        min_val = self.data[0]
        max_val = self.data[0]

        for i from 0 <= i < self.length:
            if min_val > self.data[i]:
                min_val = self.data[i]
            if max_val < self.data[i]:
                max_val = self.data[i]
        
        self.minimum = min_val
        self.maximum = max_val		   


