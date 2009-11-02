"""
Tests for classes in the solver_component module.
"""

# standard imports
import unittest

# local import
from pysph.solver.solver_component import SolverComponent, ComponentManager
from pysph.solver.entity_base import EntityBase, Solid, Fluid
from pysph.solver.entity_types import *


################################################################################
# `TestSolverComponent` class.
################################################################################ 
class TestSolverComponent(unittest.TestCase):
    """
    Tests for the SolverComponent class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        c = SolverComponent()
        self.assertEqual(c.information.get_dict(
                SolverComponent.PARTICLE_PROPERTIES_WRITE), {})
        self.assertEqual(c.information.get_dict(
                SolverComponent.PARTICLE_PROPERTIES_READ), {})
        self.assertEqual(c.information.get_dict(
                SolverComponent.PARTICLE_FLAGS), {})
        self.assertEqual(c.information.get_dict(
                SolverComponent.ENTITY_PROPERTIES), {})
        self.assertEqual(c.information.get_dict(
                SolverComponent.PARTICLE_PROPERTIES_PRIVATE), {})
        self.assertEqual(c.information.get_dict(
                SolverComponent.INPUT_TYPES), {})
        self.assertEqual(c.information.get_dict(
                SolverComponent.ENTITY_NAMES), {})
        self.assertEqual(c.information.get_dict(
                SolverComponent.OUTPUT_PROPERTIES), {})

    def test_filter_entity(self):
        """
        Tests the filter_entity function.
        """
        e = EntityBase('e', {'a':2.0, 'b':5.0, 'c':6.0})

        c = SolverComponent()
        input_types = c.information.get_dict(SolverComponent.INPUT_TYPES)

        # make the component accept Fluids only.
        input_types[EntityTypes.Entity_Fluid] = None
        self.assertEqual(c.filter_entity(e), True)
        
        # remove any type requirements.
        # now the entity should be accepted.
        input_types.pop(EntityTypes.Entity_Fluid)
        self.assertEqual(c.filter_entity(e), False)

        e = Solid()
        input_types[EntityTypes.Entity_Solid] = None
        self.assertEqual(c.filter_entity(e), False)

        #  Solid should be accepted when the component has a type requirement of
        #  Base.
        input_types.pop(EntityTypes.Entity_Solid)
        input_types[EntityTypes.Entity_Base] = None
        self.assertEqual(c.filter_entity(e), False)

        input_types.pop(EntityTypes.Entity_Base)
        input_types[EntityTypes.Entity_Fluid] = None
        self.assertEqual(c.filter_entity(e), True)

        # now add some named requirements.
        entity_names = c.information.get_dict(SolverComponent.ENTITY_NAMES)

        entity_names[EntityTypes.Entity_Fluid] = ['f1']
        entity_names[EntityTypes.Entity_Solid] = ['s1']
        
        # clear all property requirements.
        input_types.clear()
        
        e1 = Solid('s1')
        e2 = Fluid('f1')
        e3 = Fluid('f2')
        
        self.assertEqual(c.filter_entity(e1), False)
        self.assertEqual(c.filter_entity(e2), False)
        self.assertEqual(c.filter_entity(e3), True)

################################################################################
# `DummyComponent1` class.
################################################################################
class DummyComponent1(SolverComponent):
    """
    Component for testing the ComponentManager class.
    """
    def __init__(self, name=''):
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

################################################################################
# `DummyComponent2` class.
################################################################################
class DummyComponent2(SolverComponent):
    """
    Component for testing the ComponentManager class.
    """
    def __init__(self, name=''):
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

################################################################################
# `DummyComponent2` class.
################################################################################
class DummyComponent3(SolverComponent):
    """
    Component for testing the ComponentManager class.
    """
    def __init__(self, name=''):
        """
        Constructor.
        """
        # add read property requirement
        rp = self.information.get_dict(self.PARTICLE_PROPERTIES_READ)
        rp[EntityTypes.Entity_Fluid] = ['b', 'd']

        rp[EntityTypes.Entity_Solid] = ['a', 'b', 'c', 'd']

################################################################################
# `TestComponentManager` class.
################################################################################
class TestComponentManager(unittest.TestCase):
    """
    Tests for the ComponentManager class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        cm = ComponentManager()
        self.assertEqual(cm.component_dict, {})
        self.assertEqual(cm.information.get_dict(
                SolverComponent.PARTICLE_PROPERTIES_PRIVATE), {})
        self.assertEqual(cm.information.get_dict(
                SolverComponent.PARTICLE_PROPERTIES_WRITE), {})
        self.assertEqual(cm.information.get_dict(
                SolverComponent.PARTICLE_PROPERTIES_PRIVATE), {})
        self.assertEqual(cm.information.get_dict(
                ComponentManager.ENTITY_PROPERTIES), {})
        self.assertEqual(cm.information.get_dict(
                ComponentManager.PARTICLE_PROPERTIES), {})

    def test_add_component(self):
        """
        Tests the add_component function.
        """
        c1 = DummyComponent1('c1')
        c2 = DummyComponent2('c2')
        c3 = DummyComponent3('c3')

        cm = ComponentManager()
        cm.add_component(c1)

        # make sure the component has been added.
        self.assertEqual(cm.component_dict['c1'], (c1, False))
        # make sure the property requirements have been updated.
        particle_props = cm.information.get_dict(cm.PARTICLE_PROPERTIES)

        solid_props = particle_props[EntityTypes.Entity_Solid]
        check_particle_properties(solid_props,
                                  ['a', 'b', 'f2'],
                                  ['double', 'double', 'int'],
                                  [1.0, 2.0, 5])

        fluid_props = particle_props[EntityTypes.Entity_Fluid]
        check_particle_properties(fluid_props, 
                         ['c', 'd', 'e', 'f', 'f1'],
                         ['double', 'double', 'double', 'double', 'int'],
                         [None, None, 10.0, 11.0, 4])
        
        # add a DummyComponent3, which will be accepted for sure.
        cm.add_component(c3, True)
        self.assertEqual(cm.component_dict['c3'], (c3, True)) 
        # solid properties should contain the extra array 'd'
        check_particle_properties(solid_props,
                                  ['a', 'b', 'f2', 'd', 'c'],
                                  ['double', 'double', 'int', 'double', 'double'],
                                  [1.0, 2.0, 5, None, None])

        # fluid properties should contain the extra array 'b'
        check_particle_properties(fluid_props, 
                         ['b', 'c', 'd', 'e', 'f', 'f1'],
                         ['double', 'double', 'double', 'double', 'double', 'int'],
                         [None, None, None, 10.0, 11.0, 4])

        # now try adding a component with conflicting requirements.
        cm.add_component(c2)
        
        # the component should not have been added. And the arrays should not
        # have changed.
        self.assertEqual(cm.component_dict.has_key('c2'), False)
        # both properties should remain the same.
        check_particle_properties(solid_props,
                                  ['a', 'b', 'f2', 'd', 'c'],
                                  ['double', 'double', 'int', 'double', 'double'],
                                  [1.0, 2.0, 5, None, None])
        check_particle_properties(fluid_props, 
                         ['b', 'c', 'd', 'e', 'f', 'f1'],
                         ['double', 'double', 'double', 'double', 'double', 'int'],
                         [None, None, None, 10.0, 11.0, 4])

def check_particle_properties(prop_dict, prop_names, data_types, default_vals):
    """
    Checks if prop_dict has the names in prop_names and the required default
    values and data types.
    """
    # make sure we have the exact same number of properties.
    assert len(prop_dict.keys()) ==  len(prop_names)
    for i in range(len(prop_names)):
        prop = prop_dict[prop_names[i]]
        assert prop['default'] == default_vals[i]
        assert prop['data_type'] == data_types[i]
            
if __name__ == '__main__':
    import logging
    logger = logging.getLogger()
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    unittest.main()
