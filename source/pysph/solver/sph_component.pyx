"""
Base class for components doing some SPH summation.
"""

# logger imports
import logging
logger = logging.getLogger()

# local imports
from pysph.base.kernelbase cimport KernelBase
from pysph.base.nnps cimport NNPSManager
from pysph.solver.base cimport Base
from pysph.sph.sph_calc cimport SPHBase
from pysph.sph.sph_func cimport SPHFunctionParticle
from pysph.solver.solver_base cimport SolverComponent, SolverBase, \
    ComponentManager
from pysph.solver.entity_base cimport EntityBase


################################################################################
# `SPHSourceDestMode` class.
################################################################################
cdef class SPHSourceDestMode:
    """
    Class to hold the different ways in which source and destinations of a given
    SPH component should be handled.
    """

    Group_None = 0
    Group_By_Type = 1
    Group_All = 2

    def __cinit__(self, *args, **kwargs):
        """
        Constructor.
        """
        raise SystemError, 'Do not instantiate this class'

################################################################################
# `SPHComponent` class.
################################################################################
cdef class SPHComponent(SolverComponent):
    """
    Base class for components doing SPH summations.
    """
    def __cinit__(self, 
                  str name='',
                  SolverBase solver=None,
                  ComponentManager component_manager=None,
                  list entity_list=[],
                  NNPSManager nnps_manager=None,
                  KernelBase kernel=None,
                  list source_list=[],
                  list dest_list=[],
                  bint source_dest_setup_auto=True,
                  int source_dest_mode=SPHSourceDestMode.Group_None,
                  *args, **kwargs):
        """
        Constructor.
        """
        self.source_dest_mode = source_dest_mode
        self.source_dest_setup_auto = source_dest_setup_auto
        
        self.source_types = []
        self.dest_types = []

        self.source_list = []
        self.dest_list = []

        self.source_list[:] = source_list
        self.dest_list[:] = dest_list

        self.sph_calcs = []

        if kernel is not None:
            self.kernel = kernel
        else:
            if solver is not None:
                self.kernel = self.solver.kernel

        if solver is not None:
            self.nnps_manager = solver.nnps_manager
        else:
            self.nnps_manager = nnps_manager

        self.sph_class = SPHBase
        self.sph_func_class = None

    cpdef int setup_component(self) except -1:
        """
        """
        logger.info('SPHComponent : setting up %s'%(self.name))
        logger.info('Component object : %s'%(self))
                    
        if self.setup_done == True:
            return 0

        self.sph_calcs[:] = []
        
        self._setup_sources_dests()

        self._setup_sph_objs()
        
        self.setup_done = True

    cpdef _setup_sources_dests(self):
        """
        From the input entity_list and the source_types,
        dest_types, make two lists - sources and dests.

        This is done only if source_dest_setup_auto is True

        """
        if self.source_dest_setup_auto == False:
            return

        if len(self.source_list) > 0 or len(self.dest_list) > 0:
            logger.warn('source_list/dest_list not empty')
            logger.warn('clearing')

        self.source_list[:] = []
        self.dest_list[:] = []

        for e in self.entity_list:
            logger.debug('Adding entity : %s'%(e.name))
            if e.is_type_included(self.source_types):
                self.source_list.append(e)
            if e.is_type_included(self.dest_types):
                self.dest_list.append(e)

        logger.info('(%s)source entities : %s'%(self.name, self.source_list))
        logger.info('(%s)dest entities : %s'%(self.name, self.dest_list))

    cpdef _setup_sph_objs(self):
        """
        """
        if self.source_dest_mode == SPHSourceDestMode.Group_None:
            self._setup_sph_group_none(self.dest_list, self.sph_class,
                                       self.sph_func_class)
        elif self.source_dest_mode == SPHSourceDestMode.Group_By_Type:
            self._setup_sph_group_by_type(self.source_list, self.dest_list,
                                          self.sph_class, self.sph_func_class)
        elif self.source_dest_mode == SPHSourceDestMode.Group_All:
            self._setup_sph_group_all(self.source_list, self.dest_list, 
                                      self.sph_class, self.sph_func_class)
        else:
            self.setup_sph_objs(self.source_list, self.dest_list,
                                self.sph_class, self.sph_func_class)            

    cpdef _setup_sph_group_none(self, list dest_list, type sph_class, type
                                sph_func_class):
        """
        For each entity in the destination list, create one sph calc, and one
        sph functionfor the SPH summation. For each sph calc and function the
        entity will remain the source and destination.

        """
        
        for e in dest_list:
            parr = e.get_particle_array()
            
            if parr is None:
                msg = 'Entity %s does not have a particle array'%(e.name)
                logger.error(msg)
                raise AttributeError, msg
            
            func = sph_func_class(source=parr, dest=parr)
            
            self.setup_sph_function(e, e, func)

            calc = sph_class(sources=[parr], dest=parr, kernel=self.kernel,
                             sph_funcs=[func],
                             nnps_manager=self.nnps_manager)

            self.setup_sph_summation_object([e], e, calc)

            self.sph_calcs.append(calc)

    cpdef _setup_sph_group_by_type(self, list source_list, list dest_list, type
                                   sph_class, type sph_func_class):
        """
        Creates sph objects where sources have to be of same type as the
        destination for an sph summation. 
        
        """
        for d in dest_list:
            d_parr = d.get_particle_array()
            
            if d_parr is None:
                msg = 'Entity %s does not have a particle array'%(d.name)
                logger.error(msg)
                raise AttributeError, msg

            func_list = []
            s_list = []
            s_parr_list = []
            s_parr = None
    
            for s in source_list:
                if not s.is_a(d.type):
                    continue

                s_parr = s.get_particle_array()
                    
                if s_parr is None:
                    msg = 'Entity %s does not have a particle array'%(s.name)
                    logger.error(msg)
                    raise AttributeError, msg
        
                s_list.append(s)
                s_parr_list.append(s_parr)

                func = sph_func_class(source=s_parr, dest=d_parr)
                self.setup_sph_function(s, d, func)
                func_list.append(func)

            calc = sph_class(sources=s_parr_list, dest=d_parr,
                             kernel=self.kernel, sph_funcs=func_list,
                             nnps_manager=self.nnps_manager)

            self.setup_sph_summation_object(source_list=s_list, dest=d,
                                            sph_sum=calc)
            self.sph_calcs.append(calc)

    cpdef _setup_sph_group_all(self, list source_list, list dest_list, type
                               sph_class, type sph_func_class):
        """
        Creates sph objects when all the sources are to be considered for each
        destination.

        """
        for d in dest_list:
            d_parr = d.get_particle_array()
            
            if d_parr is None:
                msg = 'Entity %s does not have a particle array'%(d.name)
                logger.error(msg)
                raise AttributeError, msg

            func_list = []
            s_list = []
            s_parr_list = []
            s_parr = None
    
            logger.debug('Source list now : %s'%(source_list))

            for s in source_list:
                s_parr = s.get_particle_array()
                    
                if s_parr is None:
                    msg = 'Entity %s does not have a particle array'%(s.name)
                    logger.error(msg)
                    raise AttributeError, msg
        
                s_list.append(s)
                s_parr_list.append(s_parr)

                func = sph_func_class(source=s_parr, dest=d_parr)
                self.setup_sph_function(s, d, func)
                func_list.append(func)

            calc = sph_class(sources=s_parr_list, dest=d_parr,
                             kernel=self.kernel, sph_funcs=func_list,
                             nnps_manager=self.nnps_manager)

            self.setup_sph_summation_object(source_list=s_list, dest=d,
                                            sph_sum=calc)
            self.sph_calcs.append(calc)

    cpdef setup_sph_function(self, EntityBase source, EntityBase dest,
        SPHFunctionParticle sph_func):
        """
        Callbacks to perform any component specific initialization of the
        created sph function. 

        This callback is made soon after the sph function is instantiated.

        """
        pass

    cpdef setup_sph_summation_object(self, list source_list, EntityBase dest,
                                     SPHBase sph_sum):
        """
        Callback to perform any component specific initialization of the created
        sph summation object.

        This callback is made soon after the sph summation object has been
        instantiated.
        """
        pass

    cpdef setup_sph_objs(self, list source_list, list dest_list, type sph_class,
                         type sph_func_class):
        """
        Implement this function if you want a different handling of source and
        dest objects that those provided by this class.
        """
        raise NotImplementedError, 'SPHComponent::setup_sph_objs'

    cdef int compute(self) except -1:
        """
        """
        raise NotImplementedError, 'SPHComponent::compute'
################################################################################
# `PYSPHComponent` class.
################################################################################
cdef class PYSPHComponent(SPHComponent):
    """
    Component to implement SPH components from pure python.
    """
    def __cinit__(self, str name='',
                  SolverBase solver=None,
                  ComponentManager component_manager=None,
                  list entity_list=[],
                  NNPSManager nnps_manager=None,
                  KernelBase kernel=None,
                  list source_list=[],
                  list dest_list=[],
                  bint source_dest_setup_auto=True,
                  int source_dest_mode=SPHSourceDestMode.Group_None,
                  *args, **kwargs):
        pass

    def __init__(self, str name='',
                 SolverBase solver=None,
                 ComponentManager component_manager=None,
                 list entity_list=[],
                 NNPSManager nnps_manager=None,
                 KernelBase kernel=None,
                 list source_list=[],
                 list dest_list=[],
                 bint source_dest_setup_auto=True,
                 int source_dest_mode=SPHSourceDestMode.Group_None,
                 *args, **kwargs):
        pass

    cpdef int py_compute(self) except -1:
        """
        Function where the core logic of a python component is to be implemented.
        """
        raise NotImplementedError, 'UserDefinedComponent::py_compute'

    cdef int compute(self) except -1:
        """
        Call the py_compute method.
        """
        return self.py_compute()
