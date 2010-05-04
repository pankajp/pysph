"""
Base class for components doing some SPH summation.
"""

# local imports
from pysph.base.kernels cimport MultidimensionalKernel
from pysph.sph.calc cimport SPHCalc
from pysph.sph.sph_func cimport SPHFunctionParticle

from entity_base cimport EntityBase

cdef class SPHSourceDestMode:
    """
    Class to hold the different ways in which source and destinations of a given
    SPH component should be handled.

    """
    pass

cdef class SPHComponent:
    """
    An SPH solver consists of an sph_calc object handling the summations
    between the source and destination particle managers/entities. 

    The SPHComponent generalizes this concept when multiple entities are 
    present. For a simulation with multiple entities, the SPHComponent 
    creates an sph_calc object for each destination, source pair, based 
    on the user specified method for grouping them.

    Data Attributes:
    ----------------
    calcs -- The list of sph_calc's. One for each destination-source pair

    src_types -- Accepted types for the sources.
    src -- The list of source

    dst_types -- Accepted types for the destinations.
    dst -- The list of destinations

    _auto_setup -- Setup the sph_calc's automatically?
    _mode -- The mode of grouping

    nnps_manager -- The nnps's to use for the calc's

    sph -- The type of sph_calc
    sph_func -- The sph_eval function to use

    """

    # list of SPHCalc objects created for this component.
    cdef public list calcs
    
    # list of sources.
    cdef public list srcs
    
    # list of dests.
    cdef public list dsts

    # list of types accepted as sources
    cdef public list src_types

    # list of types accepted as dests.
    cdef public list dst_types

    # indcates the way to use the src and entity_list to create the sph
    # functions and calculators.
    cdef public int _mode

    cdef public list entity_list

    # the kernel to use.
    cdef public MultidimensionalKernel kernel

    # The sph_calc to use
    cdef public type sph_calc

    #The sph_eval to use
    cdef public type sph_func

    #Read, write and update properties.
    cdef public list reads
    cdef public list writes 
    cdef public list updates

    cdef public bool setup_done

    ######################################################################
    #Member functions
    ######################################################################
    cdef int compute(self) except -1

    #Setup src's and dst's
    cpdef _setup_entities(self)    

    #Setup SPH
    cpdef _setup_sph(self)

    #Group by None
    cpdef _group_by_none(self, list dst, type sph_calc,
                         type sph_eval)

    #Group by type
    cpdef _group_by_type(self, list src, list dst, type
                         sph_calc, type sph_eval)

    #Group by All
    cpdef _group_all(self, list src, list dst, type
                     sph_calc, type sph_eval)

    #Group by ?
    cpdef _group(self, list src, list dst, type sph_calc,
                 type sph_eval)

    cpdef setup_sph_function(self, EntityBase source, EntityBase dest,
                             SPHFunctionParticle sph_func)

    cpdef setup_sph_summation_object(self, list src, EntityBase dest,
                             SPHCalc sph_sum)

    cpdef int setup_component(self)

#############################################################################
