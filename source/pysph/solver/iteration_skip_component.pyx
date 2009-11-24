"""
Include classes to enalbe execution of certain components every few iterations.
"""

import logging
logger = logging.getLogger()

# local import
from pysph.solver.base cimport Base
from pysph.solver.solver_base cimport *


################################################################################
# `ComponentIterationSpec` class.
################################################################################
cdef class ComponentIterationSpec(Base):
    """
    Holds information about a component to be executed in an
    IterationSkipComponent.
    """
    def __cinit__(self, SolverComponent component=None, int skip_iteration=0, *args,
                  **kwargs):
        """
        Constructor.
        """
        self.component = component

        if skip_iteration <= 0:
            skip_iteration = 1

        self.skip_iteration = skip_iteration

    def __init__(self, SolverComponent c=None, int skip_iteration=0, *args,
                 **kwargs):
        """
        Python constructor.
        """
        pass

    
cdef class IterationSkipComponent(SolverComponent):
    """
    Class to enable execution of certain components every few iterations.
    """
    def __cinit__(self, SolverBase solver=None,
                  ComponentManager component_manager=None,
                  list entity_list=[],
                  *args, **kwargs):
        """
        Constructor.
        """
        self.component_spec_list = []

    def __init__(self, SolverBase solver=None,
                 ComponentManager component_manager=None,
                 list entity_list=[],
                 *args, **kwargs):
        """
        Python constructor.
        """
        SolverComponent.__init__(self, solver=solver,
                                 component_manager=component_manager,
                                 entity_list=entity_list, *args, **kwargs)
        

    cpdef add_component(self, SolverComponent c, int skip_iteration=1):
        """
        Adds a new component to the skip iteration component.
        """
        cinfo = ComponentIterationSpec(component=c,
                                       skip_iteration=skip_iteration)

        
        self.component_spec_list.append(cinfo)

    cpdef int setup_component(self) except -1:
        """
        """

        if self.setup_done == True:
            return 0

        if self.solver is None:
            raise SystemError, 'Solver is None'

        self.setup_done = True

        return 0

    cdef int compute(self) except -1:
        """
        Call compute of each component according to its skip iteration.
        """
        self.setup_component()

        cdef SolverComponent c

        for cinfo in self.component_spec_list:
            if (self.solver.current_iteration % cinfo.skip_iteration) == 0:
                logger.info('Calling component %s'%(cinfo.component.name))
                c = cinfo.component
                c.compute()
                
