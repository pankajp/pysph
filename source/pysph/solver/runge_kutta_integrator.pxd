"""
Module contains various basic RK integrators.
"""

# local imports
from pysph.solver.integrator_base cimport *

cdef class RK2TimeStepSetter(SolverComponent):
    """
    Component to set the time step.
    """
    cdef Integrator integrator
    cdef TimeStep time_step
    cdef int step_num

cdef class RK2Integrator(Integrator):
    """
    Runge Kutta 2 Integrator.
    """
    cdef public dict _stepper_info

    cdef public dict _step1_default_steppers
    cdef public dict _step2_default_steppers

    cdef public int step_being_setup

    cpdef int setup_component(self) except -1
    cpdef int update_property_requirements(self) except -1

cdef class RK2SecondStep(ODEStepper):
    """
    Stepper to perform the second step of a runge kutta 2 integrator.
    """
    pass
