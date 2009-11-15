"""
Contains base classes for all integrators.
"""

# local imports
from pysph.solver.base cimport Base
from pysph.solver.solver_base cimport SolverComponent
from pysph.solver.entity_base cimport EntityBase
from pysph.solver.time_step cimport TimeStep

# forward declarations.
cdef class Integrator

################################################################################
# `ODEStepper` class.
################################################################################
cdef class ODEStepper(SolverComponent):
    """
    Class to step a given property by a given time step.
    """
    # name of the property that is being stepped.
    cdef public str prop_name

    # names of properties that have to be stepped
    cdef public list integral_names

    # names of arrays where the values of the next step will be stored.
    cpdef public list next_step_names

    # names of properties representing the 'rate' of change of properties that
    # have to be stepped.
    cdef public list integrand_names

    # the time_step object to obtain the time step.
    cdef public TimeStep time_step
    
    cpdef int setup_component(self) except -1
    cdef int compute(self) except -1
    cpdef set_properties(self, str prop_name, list integrands, list integrals)

################################################################################
# `PyODEStepper` class.
################################################################################
cdef class PyODEStepper(ODEStepper):
    """
    Class to implement some steppers in pure python if needed.
    """
    cdef int compute(self) except -1
    cpdef int py_compute(self) except -1

################################################################################
# `Integrator` class.
################################################################################
cdef class Integrator(SolverComponent):
    """
    Base class for all integrators. Integrates a set of given properties.
    """
    # the final list of components that will be executed at every call to
    # compute().
    cdef public list execute_list
    
    # the dimension of the velocity and position vectors.
    cpdef public int dimension

    # the time step to use for stepping.
    cpdef public TimeStep curr_time_step

    # add an entity whose properties have to be integrated.
    cpdef add_entity(self, EntityBase entity)

    # add a component to be executed before integration of this property.
    #cpdef add_component(self, str property, str comp_name, bint pre_step=*)
    
    # add a component to be executed before stepping of a particular property.
    cpdef add_pre_step_component(self, str comp_name, str property_name=*)

    # add a component to be executed after a particular property has been
    # stepped. 
    cpdef add_post_step_component(self, str comp_name, str property_name=*)
    
    # add a component to be exectued before integration of any property is done.
    cpdef add_pre_integration_component(self, str comp_name, bint
                                        at_tail=*)
    # add a component to be executed after integration has been done - for
    # multi-step components this should add the component after the steps are
    # over.
    cpdef add_post_integration_component(self, str comp_name, bint at_tail=*)
    
    # set the order in which properties should be integrated.
    cpdef set_integration_order(self, list order)
    
    cdef int compute(self) except -1
    cdef int _integrate(self) except -1
    
    # setup the component once prior to execution.
    cpdef int setup_component(self) except -1

    # add a new property to be integrated along with arrays representing the
    # properties. 
    cpdef add_property(self, str prop_name, list integrand_arrays, list
                       integral_arrays, list entity_types=*, dict steppers=*)

    cpdef set_dimension(self, int dimension)

    # add an entity type for inclusion in integration of a given property.
    cpdef add_entity_type(self, str prop_name, int entity_type)
    cpdef remove_entity_type(self, str prop_name, int entity_type)

    # updates internal data structures about property requirements.
    cpdef int update_property_requirements(self) except -1    
