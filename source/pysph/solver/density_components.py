"""
Module containing various density components.
"""

# standard imports


# local imports
from pysph.solver.solver_component cimport SolverComponent


class SPHDensitySummationComponent(SolverComponent):
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
        self.entity_list = set(entity_list)

        # By default this component will accept only fluids or derived classes
        # as input.
        self.add_entity_type(EntityTypes.Entity_Fluid)

    def setup_component(self):
        """
        Sets up the component prior to execution.
        """
        pass

    def update_property_requirements(self):
        """
        Update the property requirements of the component.
        """
        pass

    def add_entity(self, EntityBase entity):
        """
        If entity is not filtered add it to the entity list.
        """
        if self.filter_entity(entity) is False:
            self.entity_list.add(entity)
