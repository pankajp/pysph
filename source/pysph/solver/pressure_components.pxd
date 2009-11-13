"""
Module to hold components to compute pressure.
"""

# standard imports


# local imports
from pysph.solver.solver_base cimport SolverComponent
from pysph.solver.speed_of_sound cimport SpeedOfSound

cdef class TaitPressureComponent(SolverComponent):
    """
    Component to compute pressure using the Tait equation.
    """
    cdef double gamma
    cdef SpeedOfSound speed_of_sound

    cdef int compute(self) except -1
    cpdef int setup_component(self) except -1
