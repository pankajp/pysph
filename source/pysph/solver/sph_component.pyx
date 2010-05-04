"""
Base class for components doing some SPH summation.
"""

################################################################################
# `SPHSourceDestMode` class.
################################################################################
cdef class SPHSourceDestMode:
    """
    Class to hold the different ways in which source and destinations of a given
    SPH component should be handled.

    Example: Group by None
    ----------------------
    If the group mode is None, an sph_calc object is created for each 
    destination entity, which behaves as a source as well. 
    This essentially is useful to model a scenario when all entities 
    influence themselves. An example would be a shock tube simulation
    with free boundaries.

    Example: Group by Type
    ----------------------
    Consider again the shock tube problem. The boundary and domain
    are both of type `Fluid`, but no force is computed on the boundary 
    particles. This type of grouping would setup a calc for the 
    force computation by the boundary on the domain.

    Example: Group All
    ------------------
    Consider our old friend again. The shock tube problem.
    Just create the domain fluid as the destination entity and 
    all other boundary entities and the domain fluid as the sources. 

    """

    byNone = 0
    byType = 1
    byAll = 2

    def __cinit__(self, *args, **kwargs):
        """ Constructor. """
        raise SystemError, 'Do not instantiate this class'

################################################################################
# `SPHComponent` class.
################################################################################
cdef class SPHComponent:
    """
    An SPH solver consists of an sph_calc object handling the summations
    between the source and destination particle managers/entities. 

    The SPHComponent generalizes this concept when multiple entities are 
    present. For a simulation with multiple entities, the SPHComponent 
    creates an sph_calc object for each destination, source pair, based 
    on the user specified method for grouping them.

    """

    #Defined in the .pxd file
    #cdef public list calcs
    #cdef public list srcs
    #cdef public list dsts
    #cdef public list src_types
    #cdef public list dst_types
    #cdef public int _mode
    #cdef public MultidimensionalKernel kernel
    #cdef public type sph_calc
    #cdef public type sph_func

    #cdef public list entity_list
    #cdef public list nbrloc_list

    def __init__(self, 
                  str name='',
                  list entity_list=[],
                  MultidimensionalKernel kernel=None,
                  int _mode=SPHSourceDestMode.byNone,
                  *args, **kwargs):
        """
        Parameters:
        -----------
        name -- String literal for identifying the Component
        entity_list -- The list of entities to manage
        kernel -- The kernel used for the calcs
        _mode -- The mode by which the calcs are setup
        
        Defaults:
        ------
        srcs -- []
        dsts -- []
        calcs -- []
        nbrloc_list -- []
        sph_calc = SPHCalc
        writes -- ['tmpx', 'tmpy', 'tmpz']

        Notes:
        ------
        If a component manager is being used, set the entity_list and 
        the nbrloc_list of the component to that of the manager's, before
        invoking setup_component.

        """
        self.writes = ['tmpx', 'tmpy', 'tmpz']
        self._mode = _mode
        
        self.srcs = []
        self.dsts = []

        self.calcs = []

        self.kernel = kernel
        self.entity_list = entity_list

        self.sph_calc = SPHCalc

        #Default parameters
        self.nbrloc_list = []
        
    cpdef int setup_component(self) except -1:
        """
        """
        if self.setup_done == True:
            return 0

        self.calcs = []
        
        self._setup_entities()

        self._setup_sph()
        
        self.setup_done = True

    cpdef _setup_entities(self):
        """ Reset the source_list and dest_list to empty lists before 
        populating them with entities from the entity list.

        The accepted types for the sources and destinations are 
        defined in the attributes src_types and dst_types.

        """
        self.srcs[:] = []
        self.dsts[:] = []

        #Add each entity from the list to either source or destination
        for e in self.entity_list:
            if e.is_type_included(self.src_types):
                self.srcs.append(e)
            if e.is_type_included(self.dst_types):
                self.dsts.append(e)

    cpdef _setup_sph(self):
        """ Setup the sph_calcs based on the groupping mode. """
        #Group by None
        if self._mode == SPHSourceDestMode.byNone:
            self._group_by_none(self.dsts, self.sph_calc,
                              self.sph_func)

        #Group by Type
        elif self._mode == SPHSourceDestMode.byType:
            self._group_by_type(self.srcs, self.dsts,
                              self.sph_calc, self.sph_func)
        
        #Group by All
        elif self._mode == SPHSourceDestMode.byAll:
            print 'Groping'
            self._group_all(self.srcs, self.dsts, 
                            self.sph_calc, self.sph_func)
        #Group by ?
        else:
            self._group(self.srcs, self.dsts,
                        self.sph_calc, self.sph_func)            

    cpdef _group_by_none(self, list dst, type sph_calc,
                         type sph_eval):
        """ This method of grouping is useful when each destination entity
        is influenced by itself. Like the fluid under self gravitation.

        """        
        for e in dst:
            parr = e.get_particle_array()
            
            #No particle array? Hahahahahahaha......Error!!!
            if parr is None:
                msg = 'Entity %s does not have a particle array'%(e.name)
                raise AttributeError, msg
            
            func = sph_eval(source=parr, dest=parr)            
            self.setup_sph_function(e, e, func)

            calc = sph_calc(sources=[parr], dest=parr, kernel=self.kernel,
                            sph_funcs=[func])

            self.setup_sph_summation_object([e], e, calc)

            #Append to the list of sph_calc's
            self.calcs.append(calc)

    cpdef _group_by_type(self, list src, list dst, type
                       sph_calc, type sph_eval):
        """ Create an sph_calc for only matching entity pairs """

        for d in dst:
            d_parr = d.get_particle_array()

            if d_parr is None:
                msg = 'Entity %s does not have a particle array'%(d.name)
                raise AttributeError, msg

            nbrloc_list = self.nbrloc_list
            func_list = []
            s_list = []
            s_parr_list = []
            s_parr = None
    
            for s in src:
                #Don't do anything if src and dst entity types mismatch
                if not s.is_a(d.type):
                    continue

                s_parr = s.get_particle_array()
                    
                if s_parr is None:
                    msg = 'Entity %s does not have a particle array'%(s.name)
                    raise AttributeError, msg
        
                #Append the source entity to the newly created list
                s_list.append(s)
                s_parr_list.append(s_parr)

                #Create the sph_eval function and append to the list
                func = sph_eval(source=s_parr, dest=d_parr)
                self.setup_sph_function(s, d, func)
                func_list.append(func)

            calc = sph_calc(srcs=s_parr_list, dest=d_parr,
                            kernel=self.kernel, sph_funcs=func_list)

            self.setup_sph_summation_object(src=s_list, dest=d,
                                            sph_sum=calc)
            self.calcs.append(calc)

    cpdef _group_all(self, list src, list dst, type
                    sph_calc, type sph_eval):

        """ Actually the simplest type of grouping. Here, each destination 
        entity is influenced by all the sources.

        """
        for d in dst:
            d_parr = d.get_particle_array()
           
            if d_parr is None:
                msg = 'Entity %s does not have a particle array'%(d.name)
                raise AttributeError, msg

            func_list = []
            s_list = []
            s_parr_list = []
            s_parr = None
    
            for s in src:
                s_parr = s.get_particle_array()
                    
                if s_parr is None:
                    msg = 'Entity %s does not have a particle array'%(s.name)
                    raise AttributeError, msg
        
                s_list.append(s)
                s_parr_list.append(s_parr)

                func = sph_eval(source=s_parr, dest=d_parr)
                self.setup_sph_function(s, d, func)
                func_list.append(func)

            calc = sph_calc(sources=s_parr_list, dest=d_parr,
                             kernel=self.kernel, sph_funcs=func_list)

            self.setup_sph_summation_object(src=s_list, dest=d,
                                            sph_sum=calc)
            self.calcs.append(calc)

    cpdef _group(self, list src, list dst, type sph_calc,
                         type sph_eval):
        """
        Implement this function if you want a different handling of source and
        dest objects that those provided by this class.
        """
        raise NotImplementedError, 'SPHComponent::setup_sph_objs'

    cpdef setup_sph_function(self, EntityBase source, EntityBase dest,
                             SPHFunctionParticle sph_func):
        """
        Callbacks to perform any component specific initialization of the
        created sph function. 

        This callback is made soon after the sph function is instantiated.

        """
        pass

    cpdef setup_sph_summation_object(self, list src, EntityBase dest,
                                     SPHCalc sph_sum):
        """
        Callback to perform any component specific initialization of the created
        sph summation object.

        This callback is made soon after the sph summation object has been
        instantiated.
        """
        pass

    cdef int compute(self) except -1:
        """
        """
        raise NotImplementedError, 'SPHComponent::compute'


    def reads_prop(self, str prop):
        """ The read properties is a list of strings specifying the 
        properties read from the source particle arrays. """

        if self.reads == None:
            self.reads = []

        self.reads.append(prop)

    def writes_prop(self, str prop):
        """ The writes attribute is a list of strings specifying the 
        properties written to of the destination particle array. """
        
        if self.writes == None:
            self.writes = []

        self.writes.append(prop)

    def updates_prop(self, str prop):
        if self.updates == None:
            self.updates = []

        self.updates.append(prop)

    def _check_particle_arrays(self):
        reads = self.reads
        writes = self.writes
        updates = self.updates

        dsts = self.dsts
        srcs = self.srcs
        
        ndst = len(dsts)
        nsrc = len(srcs)

        nread = len(reads)
        nwrite = len(writes)
        nupdate = len(updates)

        #Check for updates. Only for the destination particle array
        for i in range(ndst):
            dst = dsts[i]
            dst_parr = dst.get_particle_array()
            for j in range(nupdate):
                if not dst_parr.properties.has_key(updates[j]):
                    print 'Adding prop %s for destination' %(updates[j])
                    dst_parr.add_props([updates[j]])
            
        #Check for read. On both, the destination and source
        for i in range(nread):
            
            msg1 = 'Property %s must be defined!'%(reads[i])
            for j in range(ndst):
                dst = dsts[j]
                dst_pa = dst.get_particle_array()
                assert dst_pa.properties.has_key(reads[i]), msg1
                
            msg = 'Property %s must be defined for the source'%(reads[i])
            for j in range(nsrc):
                src = srcs[j]
                src_pa = src.get_particle_array()
                assert src_pa.properties.has_key(reads[i]), msg

        #For write properties, add if not present in destination
        for i in range(nwrite):            
            for j in range(ndst):
                dst = dsts[j]
                dst_pa = dst.get_particle_array()
                if not dst_pa.properties.has_key(writes[i]):
                    print 'Adding prop %s for destination' %(writes[i])
                    dst_pa.add_props([writes[i]])                

    def _compute(self):
        self._check_particle_arrays()
        self.compute()

################################################################################
