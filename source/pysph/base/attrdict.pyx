"""
Extension to the dictionary class to access keys as attributes.
"""

cdef class AttrDict:
    """
    Extension to the dictionary class to access keys as attributes.

    If a key that is not present is accessed as an attribute, an AttributeError
    is raised. To add a new key access the dictionary is the normal fashion
    (using the [] operator) to set the value.

    """
    def __init__(self, **kwargs):
        """
        Constructor.
        """
        self._dict = dict(**kwargs)
        
    def __getitem__(self, name):
        """
        """
        return self._dict[name]

    def __setitem__(self, key, value):
        """
        """
        self._dict[key] = value

    def __setattr__(self, key, value):
        """
        """
        if self._dict.has_key(key):
            self._dict[key] = value
        else:
            raise AttributeError, key
        
    def __getattr__(self, key):
        try:
            return self._dict[key]
        except KeyError:
            raise AttributeError, key

    def __len__(self):
        """
        """
        return len(self._dict)

    def keys(self):
        """
        """
        return self._dict.keys()

    def values(self):
        """
        """
        return self._dict.values()

    def update(self, dict key_values):
        """
        """
        return self._dict.update(key_values)

    def has_key(self, str prop_name):
        """
        """
        return self._dict.has_key(prop_name)
