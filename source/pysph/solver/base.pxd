"""
Base class for all classes in the solver module.
"""

from pysph.solver.typed_dict cimport TypedDict

cdef class Base:
    """
    Base class for all classes in the solver module.

    Any class derived from the base can store any information associated with
    it, in the "information" attribute. The key associated with a particular
    information *should* be exposed by the class as a "class attribute". This
    allows a simple method to extend the definition of various objects later.
    Another more important use of the 'information' is to allow information
    between different modules to be communicated. Such information may not be
    part of the object at all times, but use of some module may need some extra
    information to be associated with the module, which is filled in by some
    other module. If a particular information is being required almost by all
    entities, it can be made into a attribute of the class.

    The exposed keys are specified in the pyx file as class attributes.

    The information can be retrieved from the TypedDict using any of the
    Get methods of the TypedDict.

    

    """
    cdef public TypedDict information
    
