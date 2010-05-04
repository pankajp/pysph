"""
Lists the codes for various entity types.
"""

cdef class EntityTypes:
    """
    Empty class to emulate an enum for all entity types.

    Add any new entity as a class attribute.
    """
    Entity_Base = 0
    Entity_Solid = 1
    Entity_Fluid = 2

    Entity_Dummy = 100

    def __cinit__(self):
        """
        Constructor.

        We do not allow this class to be instantiated. Only the class attributes
        are directly accessed. Instantiation will raise an error.
        """
        raise SystemError, 'Do not instantiate the EntityTypes class'
