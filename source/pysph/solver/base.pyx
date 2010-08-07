"""
Base class for all classes in the solver module.
"""

#from pysph.solver.typed_dict cimport TypedDict

cdef class Base:
    """
    Base class for all classes in the solver module.
    
    The base class currently does not contain any information.
    """
    def __cinit__(self, *args, **kwargs):
        """
        Constructor.
        """
        self.information = dict()
