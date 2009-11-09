"""
Module to include dummy components used for tests etc.
"""

# local imports
from pysph.solver.integrator_base import PyODEStepper
from pysph.solver.solver_component import ComponentManager, SolverComponent
from pysph.solver.entity_types import *


################################################################################
# `YAStepper` class.
################################################################################
class YAStepper(PyODEStepper):
    """
    Yet another stepper.
    Used for tests.
    """
    identifier = 'ya_stepper'
    category = 'dummy'
    def __init__(self, name='', cm=None, entity_list=[],
                 prop_name='', integrand_arrays=[], integral_arrays=[],
                 time_step=None, *args, **kwargs): 
        """
        Constructor.
        """
        pass
################################################################################
# `DummyComponent1` class.
################################################################################
class DummyComponent1(SolverComponent):
    """
    Component for testing the ComponentManager class.
    """
    def __init__(self, name='', cm=None):
        """
        Constructor.
        """
        # add private particle property requirements
        pp = self.information.get_dict(self.PARTICLE_PROPERTIES_PRIVATE)
        # add a requirement for Entity_Solid
        pp[EntityTypes.Entity_Solid] = [{'name':'a', 'default':1.0},
                                        {'name':'b', 'default':2.0}]
        
        # add read property requirement
        rp = self.information.get_dict(self.PARTICLE_PROPERTIES_READ)
        rp[EntityTypes.Entity_Fluid] = ['c', 'd']

        # add write property requirements
        wp = self.information.get_dict(self.PARTICLE_PROPERTIES_WRITE)
        wp[EntityTypes.Entity_Fluid] = [{'name':'e', 'default':10.0},
                                        {'name':'f', 'default':11.0}]

        flags = self.information.get_dict(self.PARTICLE_FLAGS)
        flags[EntityTypes.Entity_Fluid] = [{'name':'f1', 'default':4}]
        flags[EntityTypes.Entity_Solid] = [{'name':'f2', 'default':5}]        

        # add entity property requirements
        ep = self.information.get_dict(self.ENTITY_PROPERTIES)
        ep[EntityTypes.Entity_Fluid] = [{'name':'h', 'default':0.1},
                                        {'name':'mu', 'default':None}]

        ep[EntityTypes.Entity_Solid] = [{'name':'mu', 'default':None}]

################################################################################
# `DummyComponent2` class.
################################################################################
class DummyComponent2(SolverComponent):
    """
    Component for testing the ComponentManager class.
    """
    def __init__(self, name='', cm=None):
        """
        Constructor.
        """
        # add private particle property requirements
        pp = self.information.get_dict(self.PARTICLE_PROPERTIES_PRIVATE)
        # add a requirement for Entity_Solid
        pp[EntityTypes.Entity_Solid] = [{'name':'a', 'default':1.0},
                                        {'name':'t', 'default':2.0}]
        
        # add read property requirement
        rp = self.information.get_dict(self.PARTICLE_PROPERTIES_READ)
        rp[EntityTypes.Entity_Fluid] = ['b', 'd']

        # add write property requirements
        wp = self.information.get_dict(self.PARTICLE_PROPERTIES_WRITE)
        wp[EntityTypes.Entity_Fluid] = [{'name':'e', 'default':10.0},
                                        {'name':'f', 'default':11.0},
                                        {'name':'a', 'default':None}]

        wp[EntityTypes.Entity_Solid] = [{'name':'g', 'default':10.0},
                                        {'name':'h', 'default':11.0}]

        flags = self.information.get_dict(self.PARTICLE_FLAGS)
        flags[EntityTypes.Entity_Fluid] = [{'name':'f3', 'default':4}]
        flags[EntityTypes.Entity_Solid] = [{'name':'f4', 'default':5}]        

        # add entity property requirements
        ep = self.information.get_dict(self.ENTITY_PROPERTIES)
        ep[EntityTypes.Entity_Fluid] = [{'name':'h1', 'default':0.1},
                                        {'name':'mu', 'default':1.0}]

        ep[EntityTypes.Entity_Solid] = [{'name':'g', 'default':None}]

################################################################################
# `DummyComponent2` class.
################################################################################
class DummyComponent3(SolverComponent):
    """
    Component for testing the ComponentManager class.
    """
    def __init__(self, name='', cm=None):
        """
        Constructor.
        """
        # add read property requirement
        rp = self.information.get_dict(self.PARTICLE_PROPERTIES_READ)
        rp[EntityTypes.Entity_Fluid] = ['b', 'd']

        rp[EntityTypes.Entity_Solid] = ['a', 'b', 'c', 'd']

        ep = self.information.get_dict(self.ENTITY_PROPERTIES)
        ep[EntityTypes.Entity_Solid] = [{'name':'mu', 'default':1.0}]

        ep[EntityTypes.Entity_Fluid] = [{'name':'nu', 'default':3.0}]    

        self.entity_list = []

    def add_entity(self, entity):
        """
        Add entity to the component.
        """
        if not self.filter_entity(entity):
            self.entity_list.append(entity)
