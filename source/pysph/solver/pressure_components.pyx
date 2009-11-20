"""
Module containing components that compute pressure.
"""

# logging imports
import logging
logger = logging.getLogger()

# standard imports
cimport numpy
import numpy

# local imports
from pysph.base.particle_array cimport ParticleArray
from pysph.solver.entity_types cimport EntityTypes
from pysph.solver.entity_base cimport EntityBase
from pysph.solver.fluid cimport Fluid
from pysph.solver.solver_base cimport SolverComponent, SolverBase\
    ,ComponentManager
from pysph.solver.speed_of_sound cimport SpeedOfSound

################################################################################
# `TaitPressureComponent` class.
################################################################################
cdef class TaitPressureComponent(SolverComponent):
    """
    Component to compute pressure of fluids using the Tait equation.

    **Parameters**

        - gamma - gamma value to be used.
        
    **Keyword args that are checked for**
        
        - speed_of_sound - if solver is None, this key is checked for.
    """
    def __cinit__(self, str name='', 
                  SolverBase solver=None, 
                  ComponentManager component_manager=None, 
                  list entity_list=[], 
                  double gamma=7.0, SpeedOfSound speed_of_sound=None,
                  *args, **kwargs):
        """
        Constructor.
        """
        self.gamma = gamma
        
        if solver is not None:
            self.speed_of_sound = solver.speed_of_sound
        else:
            self.speed_of_sound = speed_of_sound

        # set the accepted input types of this component.
        self.add_input_entity_type(EntityTypes.Entity_Fluid)

    cpdef int update_property_requirements(self) except -1:
        """
        Setup the property requirements of this component.
        """
        cdef dict wp = self.information.get_dict(self.PARTICLE_PROPERTIES_WRITE)
        cdef dict rp = self.information.get_dict(self.PARTICLE_PROPERTIES_READ)
        cdef dict ep = self.information.get_dict(self.ENTITY_PROPERTIES)

        wp.clear()
        rp.clear()

        # needs property/array named 'p' and will write into it.
        wp[EntityTypes.Entity_Fluid] = [{'name':'p', 'default':0.0}]
        # needs property/array named 'rho' to read values from.
        rp[EntityTypes.Entity_Fluid] = ['rho']

        # requires and entity property called 'B' for fluids.
        ep[EntityTypes.Entity_Fluid] = [{'name':'B', 'default':1.0},
                                        {'name':'rho', 'default':1000.0},
                                        {'name':'max_density_variation',
                                         'default':0.01}]
    
    cpdef int setup_component(self) except -1:
        """
        Sets up the component prior to execution.

        **Algorithm**

            - for each entity in entity_list
                compute the 'B' value using:
                    B = fluid_rho*(speed_of_sound)^2/gamma

        """
        if self.setup_done == True:
            return 0

        logger.info('Setting up component %s'%(self.name))

        err = True
        if self.speed_of_sound is None:
            if self.solver is not None:
                if hasattr(self.solver, 'speed_of_sound'):
                    self.speed_of_sound = self.solver.speed_of_sound
                    err = False
        if err:
            self.speed_of_sound = SpeedOfSound(100.)
            logger.warn('Using speed of sound of 100')
            
        cdef double fac = (
            self.speed_of_sound.value*self.speed_of_sound.value/self.gamma)

        for e in self.entity_list:
            # make sure the entity has a particle array.

            if e.get_particle_array() is None:
                msg = 'Entity %s does not provide particle array'%(e.name)
                logger.error(msg)
                raise ValueError, msg

            e.properties.B = e.properties.rho*fac
            msg = 'B for %s is %f'%(e.name, e.properties.B)
            logger.info(msg)

        self.setup_done = True

        return 0

    cdef int compute(self) except -1:
        """
        Computes pressure of the input fluid entities.
        """
        cdef int i, num_entities
        cdef Fluid entity
        cdef ParticleArray parr
        cdef numpy.ndarray rho, rho_dash
        cdef double e_rho, e_B

        # make sure component is properly setup.
        self.setup_component()

        num_entities = len(self.entity_list)
        
        for i from 0 <= i < num_entities:
            entity = self.entity_list[i]
            parr = entity.get_particle_array()

            e_rho = entity.properties.rho
            e_B = entity.properties.B
            parr.p[:] = 0.0
            parr.p[:] = numpy.divide(parr.rho, e_rho)
            parr.p[:] = numpy.power(parr.p, self.gamma)
            parr.p -= 1.0
            parr.p *= e_B
        
        return 0
