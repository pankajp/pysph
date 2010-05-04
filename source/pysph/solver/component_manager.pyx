""" Definitions for the ComponentManager """


############################################################################
#`ComponentManager` class
############################################################################
cdef class ComponentManager:
    """ A manager for the all the SPHComponents used in a simulation.

    Task:
    ----- 
    Initialize and return appropriate neighbor locators to 
    avoid duplication by individual components.

    """
    
    #Defined in the .pxd file
    #cdef public list entity_list
    #cdef public list component_list
    #cdef public list nbrloc_list
    #cdef public int ncomponents

    def __init__(self, list entity_list, list component_list = []):
        """ Constructor """
        self.entity_list = entity_list
        self.component_list = component_list
        self.ncomponents = len(component_list)
        self.nbrloc_list = []
        self._initialize()

    cpdef _initialize(self):
        """ Create the Neighbor locators (NNPS's) for each entity. """

        cdef EntityBase entity
        cdef ParticleArray entity_pa
        cdef int i

        cdef list elist = self.entity_list
        cdef int ne = len(elist)

        for i in range(ne):
            entity = elist[i]
            entity_pa = entity.get_particle_array()
            nps = NNPS(pa)
            self.nbrloc_list.append(nps)

    cpdef add_component(self, SPHComponent component):
        """ Add a given component to the component list.
        
        Note:
        -----
        No checking is done to ensure that the component already exists.

        """
        slef.component_list.append(component)
        self.ncomponents += 1
        
    cpdef NNPS get_nbr_locator(self, EntityBase entity):
        """ Return the NNPS corresponding to the particular entity.
        
        Notes:
        -----
        We assume that the entity is present in the manager's entity_list

        """
        cdef list elist = self.entity_list
        cdef list nlist = self.nbrloc_list
        cdef int idx 
        
        try:
            idx = elist.index(entity)

        except ValueError:
            raise ValueError, 'Entity %s is not in the entity_list'%(entity)

        return nlist[idx]
        
    cpdef setup_components(self):
        """ Call setup_component for each component in the list. """

        cdef list elist = self.entity_list
        cdef list clist = self.component_list
        cdef list nlist = self.nbrloc_list
        cdef int nc = self.ncomponents

        cdef EntityBase enity
        cdef SPHComponent component
        cdef int i

        for i in range(nc):
            component = clist[i]
            component.entity_list = elist
            component.nbrloc_list = nlist
            
            component.setup_component()
############################################################################
