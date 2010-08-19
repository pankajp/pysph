"""
Tests for classes in the solver_base module.
"""

# standard imports
import unittest

# local import
from pysph.solver.solver_base import SolverComponent, ComponentManager, SolverBase
from pysph.solver.entity_base import EntityBase
from pysph.solver.fluid import Fluid
from pysph.solver.solid import Solid

from pysph.solver.dummy_components import DummyComponent1, \
    DummyComponent2, DummyComponent3
from pysph.base.particle_array import ParticleArray
from pysph.solver.dummy_entities import DummyEntity

from pysph.base.cell import CellManager
from pysph.base.nnps import NNPSManager
from pysph.base.kernels import KernelBase
from pysph.solver.integrator_base import Integrator
        
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
        assert prop['type'] == data_types[i]

def check_entity_properties(prop_dict, prop_names, default_vals):
    """
    Checks if prop_dict has the names in prop_names and the required default
    values.
    """
    # make sure we have the exact same number of properties.
    assert len(prop_dict.keys()) ==  len(prop_names)
    for i in range(len(prop_names)):
        prop = prop_dict[prop_names[i]]
        val = prop['default'] == default_vals[i]
        msg = '%s != %s'%(str(prop['default']), str(default_vals[i]))
        msg += ' for property %s'%(prop_names[i])
        assert val, msg
    
###############################################################################
# `TestSolverComponent` class.
############################################################################### 
class TestSolverComponent(unittest.TestCase):
    """
    Tests for the SolverComponent class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        c = SolverComponent()

        self.assertEqual(c.setup_done, False)

        self.assertEqual(c.particle_props_read, {})
        self.assertEqual(c.particle_props_write, {})
        self.assertEqual(c.particle_props_private, {})
        self.assertEqual(c.particle_flags, {})
        self.assertEqual(c.entity_props, {})
        self.assertEqual(c.input_types, set())
        self.assertEqual(c.entity_names, {})
        
    def test_filter_entity(self):
        """
        Tests the filter_entity function.
        """
        e = EntityBase('e', {'a':2.0, 'b':5.0, 'c':6.0})

        c = SolverComponent()
        input_types = c.input_types

        # make the component accept Fluids only.
        input_types.add(Fluid)
        self.assertEqual(c.filter_entity(e), True)
        
        # remove any type requirements.
        # now the entity should be accepted.
        input_types.remove(Fluid)
        self.assertEqual(c.filter_entity(e), False)

        e = Solid()
        input_types.add(Solid)
        self.assertEqual(c.filter_entity(e), False)

        #  Solid should be accepted when the component has a type requirement of
        #  Base.
        input_types.remove(Solid)
        input_types.add(EntityBase)
        self.assertEqual(c.filter_entity(e), False)

        input_types.remove(EntityBase)
        input_types.add(Fluid)
        self.assertEqual(c.filter_entity(e), True)

        # now add some named requirements.
        entity_names = c.entity_names

        entity_names[Fluid] = set(['f1'])
        entity_names[Solid] = set(['s1'])
        
        # clear all property requirements.
        input_types.clear()
        
        e1 = Solid('s1')
        e2 = Fluid('f1')
        e3 = Fluid('f2')
        
        self.assertEqual(c.filter_entity(e1), False)
        self.assertEqual(c.filter_entity(e2), False)
        self.assertEqual(c.filter_entity(e3), False)

    def test_add_property_speification(self):
        """
        Tests functions that add property requirements to the component. 
        """
        s = SolverComponent()
        s.add_read_prop_requirement(Fluid, ['a', 'b', 'c'])
        s.add_read_prop_requirement(EntityBase, ['d', 'e', 'f'])

        rp = s.particle_props_read
        self.assertEqual(rp[Fluid], set(['a', 'b', 'c']))
        self.assertEqual(rp[EntityBase], set(['d', 'e', 'f']))

        s.add_write_prop_requirement(EntityBase, 't')
        s.add_write_prop_requirement(Fluid, 'u', -1.02)
        
        wp = s.particle_props_write
        self.assertEqual(wp[EntityBase], [{'name':'t', 'default':0.0}])
        self.assertEqual(wp[Fluid], [{'name':'u', 'default':-1.02}])

        pp = s.particle_props_private
        s.add_private_prop_requirement(EntityBase, 'g', 9.0)
        s.add_private_prop_requirement(EntityBase, 'h', 6.0)
        
        self.assertEqual(pp[EntityBase], [{'name':'g', 'default':9.0},
                                          {'name':'h', 'default':6.0}])

        fl = s.particle_flags
        s.add_flag_requirement(Solid, 'boundary', 1)
        s.add_flag_requirement(Fluid, 'real', 5)
        self.assertEqual(fl[Solid], [{'name':'boundary', 'default':1}])
        self.assertEqual(fl[Fluid], [{'name':'real', 'default':5}])

        ep = s.entity_props
        s.add_entity_prop_requirement(Solid, 'ht', 5.0)
        self.assertEqual(ep[Solid], [{'name':'ht', 'default':5.0}])
                             
    def test_entity_type_specs(self):
        """
        Tests the add_input_entity_type, remove_input_entity_type and
        set_input_entity_types functions.
        """
        s = SolverComponent()
        s.add_input_entity_type(Solid)
        s.add_input_entity_type(DummyEntity)
        
        input_types = s.input_types
        self.assertEqual(DummyEntity in input_types, True)
        self.assertEqual(Solid in input_types, True)

        s.remove_input_entity_type(Solid)

        self.assertEqual(DummyEntity in input_types, True)
        self.assertEqual(Solid in input_types, False)

        s.set_input_entity_types([Fluid])
        self.assertEqual(Fluid in input_types, True)

###############################################################################
# `TestComponentManager` class.
###############################################################################
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
        self.assertEqual(cm.particle_props, {})
        self.assertEqual(cm.entity_props, {})
        self.assertEqual(cm.property_component_map, {})
        self.assertEqual(cm.particle_props_read, {})
        self.assertEqual(cm.particle_props_write, {})
        self.assertEqual(cm.particle_props_private, {})
        
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
        self.assertEqual(cm.component_dict['c1']['component'], c1)
        self.assertEqual(cm.component_dict['c1']['notify'], False)
        # make sure the property requirements have been updated.
        particle_props = cm.particle_props
        entity_props = cm.entity_props

        solid_props = particle_props[Solid]
        check_particle_properties(solid_props,
                                  ['a', 'b', 'f2'],
                                  ['double', 'double', 'int'],
                                  [1.0, 2.0, 5])

        fluid_props = particle_props[Fluid]
        check_particle_properties(fluid_props, 
                         ['c', 'd', 'e', 'f', 'f1'],
                         ['double', 'double', 'double', 'double', 'int'],
                         [None, None, 10.0, 11.0, 4])

        # check if the entity properties were added.
        fluid_entity_props = entity_props[Fluid]
        check_entity_properties(fluid_entity_props,
                                ['h', 'mu'],
                                [0.1, None])
        solid_entity_props = entity_props[Solid]
        check_entity_properties(solid_entity_props,
                                ['mu'],
                                [None])
        
        # add a DummyComponent3, which will be accepted for sure.
        cm.add_component(c3, True)
        self.assertEqual(cm.component_dict['c3']['component'], c3)
        self.assertEqual(cm.component_dict['c3']['notify'], True)
        
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

        # check if the entity properties were added.
        fluid_entity_props = entity_props[Fluid]
        check_entity_properties(fluid_entity_props,
                                ['h', 'mu', 'nu'],
                                [0.1, None, 3.0])
        solid_entity_props = entity_props[Solid]
        check_entity_properties(solid_entity_props,
                                ['mu'],
                                [1.0])

        # now try adding a component with conflicting requirements.
        self.assertRaises(ValueError, cm.add_component, c2)
                
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

    def test_setup_entity(self):
        """
        Tests the setup_entity function.
        """
        cm = ComponentManager()
        c1 = DummyComponent1('c1')
        c2 = DummyComponent3('c2')

        cm.add_component(c1)
        cm.add_component(c2)

        e1 = Solid(particles=ParticleArray())
        e2 = Fluid(particles=ParticleArray())
        cm.setup_entity(e1)
        cm.setup_entity(e2)

        # make sure the entity has the required properties.
        self.assertEqual(e1.properties.has_key('mu'), True)
        self.assertEqual(e2.properties.has_key('h'), True)
        self.assertEqual(e2.properties.has_key('mu'), True)

        # make sure the particle properties have been added.
        parr = e2.get_particle_array()
        self.assertEqual(parr.properties.has_key('c'), True)
        self.assertEqual(parr.properties.has_key('d'), True)
        self.assertEqual(parr.properties.has_key('e'), True)
        self.assertEqual(parr.properties.has_key('f'), True)
        self.assertEqual(parr.properties.has_key('f1'), True)
        self.assertEqual(parr.properties.has_key('b'), True)

    def test_get_component(self):
        """
        Tests the get_component function.
        """
        cm = ComponentManager()
        c1 = DummyComponent1('c1')
        c2 = DummyComponent1('c2')
        
        c3 = DummyComponent3('c3')
        c4 = DummyComponent3('c4')
        
        cm.add_component(c1)
        cm.add_component(c2)
        cm.add_component(c3)
        cm.add_component(c4)
        
        self.assertEqual(cm.get_component('c1'), c1)
        self.assertEqual(cm.get_component('c2'), c2)
        self.assertEqual(cm.get_component('c3'), c3)
        self.assertEqual(cm.get_component('c4'), c4)

    def test_add_input(self):
        """
        Tests the add_input function.
        """
        cm = ComponentManager()
        
        c1 = DummyComponent3('c1')
        c2 = DummyComponent3('c2')

        cm.add_component(c1, True)
        cm.add_component(c2)

        e1 = EntityBase()
        e2 = EntityBase()

        cm.add_input(e1)
        cm.add_input(e2)

        self.assertEqual(c1.entity_list, [e1, e2])
        self.assertEqual(c2.entity_list, [])

###############################################################################
# `TestSolverBase` class.
###############################################################################
class TestSolverBase(unittest.TestCase):
    """
    Tests the SolverBase class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        s = SolverBase()
        
        self.assertEqual(s.component_manager != None, True)
        self.assertEqual(s.cell_manager != None, True)
        self.assertEqual(s.nnps_manager != None, True)
        self.assertEqual(s.kernel, None)
        #self.assertEqual(s.integrator, None)
        self.assertEqual(s.elapsed_time, 0.0)
        self.assertEqual(s.total_simulation_time, 0.0)
        self.assertEqual(s.current_iteration, 0)
        self.assertEqual(s.time_step.value, 0.0)

        cm = ComponentManager()
        cell_man = CellManager()
        nnps_manager = NNPSManager()
        kernel = KernelBase()
        integrator = Integrator()

        s = SolverBase(component_manager=cm,
                       nnps_manager=nnps_manager,
                       cell_manager=cell_man,
                       kernel=kernel,
                       integrator=integrator,
                       time_step=0.1,
                       total_simulation_time=1.0)
                       
        self.assertEqual(s.component_manager, cm)
        self.assertEqual(s.cell_manager, cell_man)
        self.assertEqual(s.nnps_manager, nnps_manager)
        self.assertEqual(s.kernel, kernel)
        self.assertEqual(s.integrator, integrator)
        self.assertEqual(s.elapsed_time, 0.0)
        self.assertEqual(s.total_simulation_time, 1.0)
        self.assertEqual(s.time_step.value, 0.1)


if __name__ == '__main__':
    import logging
    logger = logging.getLogger()
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    unittest.main()
