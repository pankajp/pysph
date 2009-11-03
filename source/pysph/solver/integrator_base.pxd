"""
Contains base classes for all integrators.
"""

# local imports
from pysph.solver.solver_component cimport SolverComponent
from pysph.solver.entity_base cimport EntityBase


################################################################################
# `Integrator` class.
################################################################################
cdef class Integrator(SolverComponent):
    """
    Base class for all integrators. Integrates a set of given properties.
    """
    # the final list of components that will be executed at every call to
    # compute().
    cdef public execute_list
    
    # list of entities whose properties have to be integrated.
    cdef public set entity_list

    # add an entity whose properties have to be integrated.
    cpdef add_entity(self, EntityBase entity)

    # add a component to be executed before integration of this property.
    cpdef add_component(self, str property, str comp_name, bint pre_step=*)
    
    # add a component to be exectued before integration of any property is done.
    cpdef add_pre_integration_component(self, str comp_name, bint
                                        at_tail=*)
    
    # set the order in which properties should be integrated.
    cpdef set_integration_order(self, list order)
    
    cdef int compute(self) except -1
    
    # setup the component once prior to execution.
    cpdef int setup_component(self) except -1
    
