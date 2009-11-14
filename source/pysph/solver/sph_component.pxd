"""
Base class for components doing some SPH summation.
"""

# local imports
from pysph.base.kernelbase cimport KernelBase
from pysph.base.nnps cimport NNPSManager
from pysph.solver.base cimport Base
from pysph.sph.sph_calc cimport SPHBase
from pysph.sph.sph_func cimport SPHFunctionParticle
from pysph.solver.solver_base cimport SolverComponent, SolverBase,\
    ComponentManager 
from pysph.solver.entity_base cimport EntityBase

cdef class SPHSourceDestMode(Base):
    """
    Class to hold the different ways in which source and destinations of a given
    SPH component should be handled.

    """
    pass

cdef class SPHComponent(SolverComponent):
    """
    """
    # list of SPHBase objects created for this component.
    cdef public list sph_calcs
    
    # list of sources.
    cdef public list source_list
    
    # list of dests.
    cdef public list dest_list

    # indicates if the source destinations are to be infered automatically from
    # the input entity list.
    cdef public bint source_dest_setup_auto

    # indcates the way to use the source_list and entity_list to create the sph
    # functions and calculators.
    cdef public int source_dest_mode

    # the kernel to use.
    cdef public KernelBase kernel

    # the nnps manager to use.
    cpdef public NNPSManager nnps_manager

    # the type of sph summation class to create for the summation.
    cdef public type sph_class
    # the type of the sph function class to create.
    cdef public type sph_func_class

    cpdef _setup_sources_dests(self)
    cpdef _setup_sph_objs(self)

    cpdef _setup_sph_group_none(self, list dest_list, type sph_class,
                                type sph_func_class)
    cpdef _setup_sph_group_by_type(self, list source_list, list dest_list, type
                                   sph_class, type sph_func_class)
    cpdef _setup_sph_group_all(self, list source_list, list dest_list, type
                               sph_class, type sph_func_class)
    cpdef setup_sph_objs(self, list source_list, list dest_list, type sph_class,
                         type sph_func_class)

    cpdef setup_sph_function(self, EntityBase source, EntityBase dest,
                             SPHFunctionParticle sph_func)
    cpdef setup_sph_summation_object(self, list source_list, EntityBase dest,
                             SPHBase sph_sum)
    
