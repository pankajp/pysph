"""
Base class for all components that perform file write operations.
"""

# local imports
from pysph.solver.solver_base cimport *



###############################################################################
# `FileWriterComponent` class.
###############################################################################
cdef class FileWriterComponent(SolverComponent):
    """
    Base class for all components that write to file.
    """
    def __cinit__(self, name='',
                  SolverBase solver=None,
                  ComponentManager component_manager=None,
                  list entity_list=[],
                  *args, **kwargs):
        """
        Constructor.
        """
        pass

    def __init__(self, name='',
                 SolverBase solver=None,
                 ComponentManager component_manager=None,
                 list entity_list=[],
                 *args, **kwargs):
        SolverComponent.__init__(self, name=name, solver=solver,
                                 component_manager=component_manager,
                                 entity_list=entity_list,
                                 *args, **kwargs)

    cpdef int setup_component(self) except -1:
        """
        """
        return 0

    cdef int compute(self) except -1:
        """
        """
        self.setup_component()

        return self.write()
    
    cpdef int write(self) except -1:
        """
        Implement the write operation here.
        """
        raise NotImplementedError, 'FileWriterComponent::write'
