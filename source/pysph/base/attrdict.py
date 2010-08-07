"""
Extension to the dictionary class to access keys as attributes.
"""

class AttrDict(dict):
    """
    Extension to the dictionary class to access keys as attributes.

    If a key that is not present is accessed as an attribute, an AttributeError
    is raised. To add a new key access the dictionary is the normal fashion
    (using the [] operator) to set the value.

    """
    
    def __setattr__(self, key, value):
        """
        """
        if dict.has_key(self, key):
            self[key] = value
        else:
            raise AttributeError, key
        
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError, key
