"""
Includes classes to enable execution of certain components every few iterations.
"""

# local imports

from pysph.solver.solver_base cimport SolverComponent

################################################################################
# `ComponentIterationSpec` class.
################################################################################
cdef class ComponentIterationSpec:
    """
    Holds information about a component to be executed in an
    IterationSkipComponent.
    """
    cdef public SolverComponent component
    cdef public int skip_iteration

################################################################################
# `IterationSkipComponent` class.
################################################################################
cdef class IterationSkipComponent(SolverComponent):
    """
    Class to enable execution of certain components every few iterations.
    """
    cdef public list component_spec_list 

    cdef int compute(self) except -1
    cpdef add_component(self, SolverComponent c, int skip_iteration=*)
