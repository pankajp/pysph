"""
Module to hold base classes for different solver components.
"""

# local imports
from pysph.solver.base cimport Base
from pysph.solver.entity_base cimport EntityBase


################################################################################
# `SolverComponent` class.
################################################################################
cdef class SolverComponent(Base):
    """
    Base class for all solver components.
    """

    # name of the component.
    cdef public str name

    # function to perform the components computation.
    cdef int compute(self) except -1

    # python wrapper to the compute function.
    cpdef int py_compute(self) except -1

    # function to filter out unwanted entities.
    cpdef bint filter_entity(self, EntityBase entity)

    # function to add entity.
    cpdef add_entity(self, EntityBase entity)

    # function to setup the component once before execution.
    cpdef int setup_component(self) except -1

################################################################################
# `UserDefinedComponent` class.
################################################################################
cdef class UserDefinedComponent(SolverComponent):
    """
    Base class to enable users to implement components in Python.
    """
    cdef int compute(self) except -1
    cpdef int py_compute(self) except -1

################################################################################
# `ComponentManager` class.
################################################################################
cdef class ComponentManager(Base):
    """
    Class to manage different components.

    **NOTES**
        - for every component, indicate if its input has to be handled by the
        component manager.
    """
    # the main dict containing all components
    cdef public dict component_dict

    # function to add this entity to all components that require their input to
    # be managed by the component manager.
    cpdef add_input(self, EntityBase entity)
    
    # adds a new component to the component manager.
    cpdef add_component(self, SolverComponent c, bint notify=*)

    # checks if property requirements for component are safe.
    cpdef bint validate_property_requirements(self, SolverComponent c) except *

    cpdef _add_particle_property(self, dict prop, int etype, str data_type=*)
    cpdef bint _check_property(
        self, dict prop, str access_mode, int etype) except *
    cpdef _update_property_component_map(
        self, str prop, str comp_name, int etype, str access_type)

    cpdef remove_component(self, str comp_name)

    cpdef SolverComponent get_component(self, str component_name)
