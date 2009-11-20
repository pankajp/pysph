"""
Base class for all components that perform file write operations.
"""

# local imports
from pysph.solver.solver_base cimport *



################################################################################
# `FileWriterComponent` class.
################################################################################
cdef class FileWriterComponent(SolverComponent):
    """
    Base class for all components that write to file.
    """
    cdef int compute(self) except -1
    cpdef int write(self) except -1
    
