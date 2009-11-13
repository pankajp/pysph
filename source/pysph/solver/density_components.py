"""
Module containing various density components.
"""

# standard imports


# local imports
from pysph.solver.solver_component cimport SolverComponent

class SPHDensityComponent(SolverComponent):
    """
    Basic SPH density computing component.
    
    This class computes density using the basic SPH summation. This is an SPH
    component, and the source and destinations for the SPH summation need to be
    figured out from the inputs. There are three ways in which this component
    can use its inputs:
    
        - entity_separate - each entity is considered separately for SPH
        summation. One SPH calculator and one sph functions is created for each
        entity, with the same entity serving as the source and destination.
        - group_types - entities of same types are grouped together. One SPH
        calc is created for each entity. That is entity is the destination there
        and all the entities of same type act as sources.
        - group_all - all entities act as sources for all other entities.


    **Members**

        - summation_mode - controls how the density is to be computed.
        - source_entity_types - list of types that can be accepted as sources in
          the density computation.
        - dest_entity_types - list of types for which density will be computed.
        
    """
    category = 'density_component'
    identifier = 'sph_density'

    entity_separate = 0
    group_types = 1
    group_all = 2

    def __init__(self, str name='', ComponentManager cm=None,
                 summation_mode=SPHDensitySummationComponent.entity_separate,
                 entity_list=[],
                 *args, **kwargs):
        """
        Constructor.
        """
        self.summation_mode = summation_mode
        self.sph_calcs = []
        self.source_entity_types = [EntityTypes.Entity_Fluid]
        self.dest_entity_types = [EntityTypes.Entity_Fluid]
        self.add_entity_type(EntityTypes.Entity_Fluid)

    def setup_component(self):
        """
        Sets up the component prior to execution.
        """
        # clear the sph calcs.
        self.sph_calcs[:] = []

        # get the source and destination lists.
        source_list , dest_list = self._get_source_dests()
        
        # generate the sph calcs
        if self.summation_mode == self.entity_separate:
            self._setup_separate_entity_sum()
        elif self.summation_mode == self.group_types:
            self._setup_group_type_sum()
        else:
            self._setup_group_all_sum()        

    def update_property_requirements(self):
        """
        Update the property requirements of the component.
        
        For every entity type that this component will accept as input, add the
        following properties:
            - read - m and h
            - write - rho, _tmp1
        """
        inp_types = self.information.get_dict(self.INPUT_TYPES)
        wp = self.information.get_dict(self.PARTICLE_PROPERTIES_WRITE)
        rp = self.information.get_dict(self.PARTICLE_PROPERTIES_READ)
        
        rp.clear()
        wp.clear()

        for key is input_types.keys():
            rp[key] = [{'name':'m', 'default':1.0},
                       {'name':'h', 'default':1.0}]
            wp[key] = [{'name':'rho', 'default':1000.0},
                       {'name':'_tmp1', 'default':0.0}]
            
    def _get_source_dests(self):
        """
        From the input entity_list and the source_entity_types,
        dest_entity_types, make two lists - sources and dests.
        """
        source_list = []
        dest_list = []
        
        for entity in self.entity_list:
            # check if this should be included in the sources
            if entity.is_type_included(self.source_entity_types):
                source_list.append(entity)
            if entity_types.is_type_included(self.dest_entity_types):
                dest_list.append(entity)

        return source_list, dest_list

    def _setup_separate_entity_sum(self, dest_list):
        """
        For each entity in the destination list, create one sph calc, and one
        sph_eval to compute the density of particles in that entity using other
        particles from the same entity.
        """
        for e in dest_list:
            parr = e.get_particle_array()
            
            if parr is None:
                msg = 'Entity %s does not have a particle array'%(e.name)
                logger.error(msg)
                raise AttributeError, msg
            
            func = SPHRho3D(parr, parr)
            calc = SPHBase([parr], parr, self.kernel, 
