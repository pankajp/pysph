"""
Classes used to pass around information among various modules, in a clean
manner. This is inspired by the vtkInformation* classes from the VTK library.
"""

cdef class TypedDict:
    """
    Class to hold information about anything.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        self.data_dict = {}
        self.type_dict = {}

    cpdef set_int(self, str key, int val):
        """
        Sets the value for key to val. 
        
        Any previous data will be deleted.
        """
        self._set_value(key, 'int', val)

    cpdef set_float(self, str key, float val):
        """
        Sets the value for key to val. 
        
        Any previous data will be deleted.
        """
        self._set_value(key, 'float', val)

    cpdef set_double(self, str key, double val):
        """
        Sets the value for key to val. 
        
        Any previous data will be deleted.
        """
        self._set_value(key, 'double', val)

    cpdef set_str(self, str key, str val):
        """
        Sets the value for key to val. 
        
        Any previous data will be deleted.
        """
        self._set_value(key, 'str', val)

    cpdef set_list(self, str key, list val):
        """
        Sets the value for key to val. 
        
        Any previous data will be deleted.
        """
        self._set_value(key, 'list', val)

    cpdef set_dict(self, str key, dict val):
        """
        Sets the value for key to val. 
        
        Any previous data will be deleted.
        """
        self._set_value(key, 'dict', val)

    cpdef set_object(self, str key, object val):
        """
        Sets the value for key to val. 
        
        Any previous data will be deleted.
        """
        self._set_value(key, 'object', val)

    cpdef int get_int(self, str key) except *:
        """
        Get the value for key.
        """
        return <int>self._get_value(key, 'int')

    cpdef float get_float(self, str key) except *:
        """
        Get the value for key.
        """
        return <float>self._get_value(key, 'float')

    cpdef double get_double(self, str key) except *:
        """
        Get the value for key.
        """
        return <double>self._get_value(key, 'double')

    cpdef str get_str(self, str key):
        """
        Get the value for key.
        """
        return <str>self._get_value(key, 'str')

    cpdef list get_list(self, str key):
        """
        Get the value for key.
        """
        return <list>self._get_value(key, 'list')

    cpdef dict get_dict(self, str key):
        """
        Get the value for key.
        """
        return <dict>self._get_value(key, 'dict')

    cpdef object get_object(self, str key):
        """
        Get the value for key.
        """
        return self._get_value(key, 'object')

    cpdef remove_key(self, str key):
        """
        Removes the said key.
        """
        self.type_dict.pop(key, None)
        self.data_dict.pop(key, None)

    cpdef int get_number_of_keys(self):
        """
        Returns the number of keys.
        """
        return len(self.data_dict.keys())

    cpdef bint has_key(self, str key, str typ=''):
        """
        Returns true if a key of the specified type is present.
        """
        if self.data_dict.has_key(key):
            if typ  == '':
                return True
            else:
                if self.type_dict[key] == typ:
                    return True
                else:
                    return False

        return False

    cpdef _set_value(self, str key, str typ, object val):
        """
        Internal function to set the value for a given key.
        """
        self.type_dict[key] = typ
        self.data_dict[key] = val

    cpdef object _get_value(self, str key, str req_typ):
        """
        Internal function to get value for the given key.
        """
        cdef str typ = self.type_dict.get(key)
        if typ == None:
            raise KeyError, 'key %s not found'%(key)

        if typ != req_typ:
            raise TypeError, 'requested %s found %s'%(req_typ, typ)

        return self.data_dict[key]
        
