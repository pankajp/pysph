"""
Components that compute time step from current solver data.
"""

# local imports
from pysph.base.carray cimport LongArray
from pysph.base.nnps cimport NNPSManager
from pysph.solver.time_step cimport TimeStep
from pysph.solver.solver_base cimport SolverComponent
from pysph.solver.speed_of_sound cimport SpeedOfSound

cdef class TimeStepComponent(SolverComponent):
    """
    Base class for all components computing time step.
    """
    cdef TimeStep time_step
    cdef double max_time_step
    cdef double min_time_step

    cpdef int update_property_requirements(self) except -1
    cpdef int setup_component(self) except -1
    cdef int compute(self) except -1

cdef class MonaghanKosTimeStepComponent(TimeStepComponent):
    """
    Component to compute time step based the paper 'Solitary waves on a cretan
    beach'.
    """
    cdef public SpeedOfSound speed_of_sound
    cdef list _sigma_arrays
    cdef public double beta
    cdef public NNPSManager nnps_manager
    cdef public list viscosity_category_names
    cdef public double viscosity_kernel_radius
    cdef public double default_visc_kernel_radius
    cdef LongArray _indices
    cdef public list nbr_locators
    
    cdef int _compute_sigmas(self, int entity_index) except -1
    cpdef double _find_viscosity_kernel_radius(self)
    cdef double _ts_range_check(self, double new_ts_value)
    cdef int compute(self) except -1

cdef class MonaghanKosForceBasedTimeStepComponent(MonaghanKosTimeStepComponent):
    """
    """
    cdef public double scale
