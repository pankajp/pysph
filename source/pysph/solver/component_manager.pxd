""" Definitions for the ComponentManager """

#solver imports
from entity_base cimport EntityBase
from sph_component cimport SPHComponent

#base imports
from pysph.base.particle_array cimport ParticleArray
from pysph.base.nnps cimport NNPS

############################################################################
#`ComponentManager` class
############################################################################
cdef class ComponentManager:

    #Data Attributes
    
    #The list of entities used in a simulation.
    cdef public list entity_list

    #The list of components required in a simulation.
    cdef public list component_list
    
    #The list of Neighbor locators.
    cdef public list nbrloc_list

    #The number of components
    cdef public int ncomponents

    #Member functions

    cpdef _initialize(self)
    cpdef add_component(self, SPHComponent component)
    cpdef setup_components(self)
    cpdef NNPS get_nbr_locator(self, EntityBase entity)
############################################################################
