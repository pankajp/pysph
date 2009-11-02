"""
Classes used to pass around information among various modules, in a clean
manner. The idea of having a separate class for storing and retrieving meta
information is inspired from VTK.


"""
cdef class TypedDict:
    """
    Class to hold information about anything.

    This is essentially an encapsulation of a dictionary, with some methods to
    clearly identify what is being set and retrieved from the dict. All
    information is maintained in a single dict, keyed on a string. We could
    later implement a faster version.
    """
    # dictionary to hold the actual data.
    cdef dict data_dict

    # dictionary to hold the that was inserted for each key.
    cdef dict type_dict

    cpdef int get_int(self, str key) except *
    cpdef double get_double(self, str key) except *
    cpdef float get_float(self, str key) except *
    cpdef str get_str(self, str key)
    cpdef list get_list(self, str key)
    cpdef dict get_dict(self, str key)
    cpdef object get_object(self, str key)

    cpdef set_object(self, str key, object val)
    cpdef set_int(self, str key, int val)
    cpdef set_float(self, str key, float val)
    cpdef set_str(self, str key, str val)
    cpdef set_double(self, str key, double val)
    cpdef set_list(self, str key, list val)
    cpdef set_dict(self, str key, dict val)

    cpdef remove_key(self, str key)
    cpdef bint has_key(self, str key, str typ=*)

    cpdef int get_number_of_keys(self)

    cpdef _set_value(self, str key, str type, object value)
    cpdef object _get_value(self, str key, str type)

