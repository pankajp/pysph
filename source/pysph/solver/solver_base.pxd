"""
Module to hold base classes for different solver components.
"""

# local imports
from pysph.solver.base cimport Base
from pysph.solver.entity_base cimport EntityBase

from pysph.base.cell cimport CellManager
from pysph.base.nnps cimport NNPSManager
from pysph.base.kernelbase cimport KernelBase
from pysph.solver.time_step cimport TimeStep
from pysph.solver.speed_of_sound cimport SpeedOfSound

# forward declaration.
cdef class ComponentManager
cdef class SolverBase

################################################################################
# `SolverComponent` class.
################################################################################
cdef class SolverComponent(Base):
    """
    Base class for all solver components.
    """

    # name of the component.
    cdef public str name

    # the solver to which this component belongs
    cdef public SolverBase solver

    # reference to the component manager.
    cdef public ComponentManager cm

    # indicates that the input entites to this component have been manually
    # added, and add_entity should not accept any more entities.
    cdef public bint accept_input_entities

    # indicates if the component is ready for execution.
    cdef public bint setup_done

    # list of input entities
    cdef public list entity_list

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

    # update the property requirements of this component
    cpdef int update_property_requirements(self) except -1

    cpdef add_entity_name(self, str name)
    cpdef remove_entity_name(self, str name)
    cpdef set_entity_names(self, list entity_names)

    cpdef add_input_entity_type(self, int etype)
    cpdef remove_input_entity_type(self, int etype)
    cpdef set_input_entity_types(self, list type_list)

    cpdef add_read_prop_requirement(self, int e_type, list prop_list)
    cpdef add_write_prop_requirement(self, int e_type, str prop_name, double
                                     default_value=*)
    cpdef add_private_prop_requirement(self, int e_type, str prop_name, double
                                       default_value=*)
    cpdef add_flag_requirement(self, int e_type, str flag_name, int
                               default_value=*)
    cpdef add_entity_prop_requirement(self, int e_type, str prop_name, double
                                      default_value=*)
    

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
        self, SolverComponent comp, dict prop, str access_mode, int etype) except *
    cpdef _update_property_component_map(
        self, str prop, str comp_name, int etype, str access_type)

    cpdef remove_component(self, str comp_name)

    cpdef SolverComponent get_component(self, str component_name)

    cpdef get_entity_properties(self, int e_type)
    cpdef get_particle_properties(self, int e_type)

    cpdef setup_entity(self, EntityBase entity)


################################################################################
# `SolverBase` class.
################################################################################
cdef class SolverBase(Base):
    """
    Base class for all solvers.
    
    This class essentially encapsulates all basic features/attributes required
    of any solver. Does not do any major processing, just a single place to hold
    lot of information required at various points in a simulation. 

    Once a few solvers have been written, some abstractions can be extracted and
    implemented in this class.

    """
    
    cdef public ComponentManager component_manager
    cdef public CellManager cell_manager
    cdef public NNPSManager nnps_manager
    cdef public KernelBase kernel

    cdef public TimeStep time_step

    cdef public double elapsed_time
    cdef public double total_simulation_time

    cdef public object integrator
    cdef public int current_iteration
    
    # a dictionary containing list of components for various categories.
    # derived classes should categories as required.
    cdef public dict component_categories

    # a dictionary containing all kernels used in the solver.
    cdef public dict kernels_used

    # a list containing all the entities involved in the simulation.
    cdef public list entity_list

    # enable/disable timing
    cdef public bint enable_timing

    # file to write output to.
    cdef public str timing_output_file

    # the timer
    cdef public object timer

    # inform the solver that this kernel is being used.
    cpdef register_kernel(self, KernelBase kernel)

    # add a new entity to be included into the simulation.
    cpdef add_entity(self, EntityBase entity)

    cpdef solve(self)
    cpdef next_iteration(self)
    
    cpdef _setup_solver(self)
    cpdef _setup_component_manager(self)
    cpdef _setup_entities(self)
    cpdef _setup_components_input(self)
    cpdef _setup_integrator(self)
    cpdef _setup_nnps(self)
    cpdef _compute_cell_sizes(self)
