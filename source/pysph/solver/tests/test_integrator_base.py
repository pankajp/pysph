"""
Tests for the integrator_base module.
"""


# standard import 
import unittest
import numpy

# local imports
from pysph.base.particle_array import ParticleArray

from pysph.solver.entity_base import EntityBase
from pysph.solver.fluid import Fluid
from pysph.solver.solid import Solid
from pysph.solver.fluid import Fluid
from pysph.solver.time_step import TimeStep
from pysph.solver.integrator_base import Integrator, ODEStepper
from pysph.solver.solver_base import ComponentManager
from pysph.solver.dummy_components import *
from pysph.solver.dummy_entities import DummyEntity
from pysph.solver.array_initializer import ArrayInitializer

def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-16."""
    return numpy.allclose(x, y, atol=1e-16, rtol=0)

class SimpleEntity(EntityBase):
    """
    Simple entity class for test purposes.
    """
    def __init__(self, name='', properties={}, particle_props={}, *args, **kwargs):
        """
        """
        self.parr = ParticleArray(name=self.name, **particle_props)

    def get_particle_array(self):
        return self.parr
    
def get_ode_step_data():
    """
    Create a particle array with data representing the case shown in image
    test_ode_step.png.
    """
    x = [-1.0, 0.0, 1.0]
    y = [1.0, 0.0, -1.0]
    
    u = [-1.0, 1.0, 0.0]
    v = [0.0, 1.0, 1.0]

    p = ParticleArray(x={'data':x}, y={'data':y}, u={'data':u}, v={'data':v})

    se = SimpleEntity()
    se.parr = p

    return se

def get_sample_integrator_setup():
    """
    Returns an integrator with some setup done. 
    Used in tests.

    The integrator component has been setup as shown in 
    integrator_test_data.dia.

    **Return Value**
    
        - component_manager
        - integrator
        - entity 1
        - entity 2
        - entity 3

    """
    # setup a component manager with some components.
    c = ComponentManager()

    # create a few components to be used in the integrator and add them to the
    # component manager.
    c1 = DummyComponent1(name='c1', component_manager=c)
    c.add_component(c1)
    c2 = DummyComponent3(name='c2', component_manager=c)
    c.add_component(c2)
    c3 = DummyComponent3(name='c3', component_manager=c)
    c.add_component(c3)
    c4 = DummyComponent1(name='c4', component_manager=c)
    c.add_component(c4)
    c5 = DummyComponent3(name='c5', component_manager=c)
    c.add_component(c5)
    c6 = DummyComponent1(name='c6', component_manager=c)
    c.add_component(c6)

    # create and setup the integrator
    i = Integrator('integrator', component_manager=c)
    prop_name = 'density'
    integrand_arrays = ['rho_rate']
    integral_arrays = ['rho']
    entity_types = [Fluid, DummyEntity]
    stepper = {'default':'euler',
               Fluid:'ya_stepper'} 
    
    i.add_property(prop_name, integrand_arrays, integral_arrays,
                   entity_types, stepper)

    # setup the entities accepted for each property stepping.
    i.add_entity_type('velocity', Fluid)
    i.add_entity_type('position', Fluid)
    i.add_entity_type('position', DummyEntity)
    
    # add pre-integration components.
    i.add_pre_integration_component(comp_name='c1')
    i.add_pre_step_component(comp_name='c2')

    # add components for the velocity stepper.
    i.add_pre_step_component(comp_name='c3', property_name='velocity')
    i.add_post_step_component(comp_name='c4', property_name='velocity')

    # add components for the position stepper.
    i.add_pre_step_component(comp_name='c5', property_name='position')

    # add a component to be executed after one complete integration.
    i.add_post_integration_component(comp_name='c6')

    # now add the integrator to the component manager.
    c.add_component(i)

    # now create the entities.
    e1 = Fluid(name='e1')
    e2 = DummyEntity('e2')
    e3 = DummyEntity('e3')

    # setup the properties of these entities
    c.setup_entity(e1)
    c.setup_entity(e2)
    c.setup_entity(e3)

    return c, i, e1, e2, e3

class TestODEStepper(unittest.TestCase):
    """
    Tests the ODEStepper class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        e = get_ode_step_data()
        ts = TimeStep(1.0)

        stepper = ODEStepper(name='', solver=None, entity_list=[e],
                             prop_name='position', integrands=['u'],
                             integrals=['x'], time_step=ts)

        self.assertEqual(stepper.entity_list, [e])
        self.assertEqual(stepper.prop_name, 'position')
        self.assertEqual(stepper.integrand_names, ['u'])
        self.assertEqual(stepper.integral_names, ['x'])
        self.assertEqual(stepper.time_step, ts)
        
    def test_setup_component(self):
        """
        Tests the setup_component function.
        """
        e = get_ode_step_data()
        e1 = EntityBase()
        ts = TimeStep(1.0)

        stepper = ODEStepper(name='', solver=None, entity_list=[e],
                             prop_name='position', integrands=['u'],
                             integrals=['x'], time_step=ts)
        stepper.setup_component()

        self.assertEqual(stepper.setup_done, True)
        self.assertEqual(stepper.next_step_names, ['x_next'])
        parr = e.get_particle_array()

        self.assertEqual(parr.properties.has_key('x_next'), True)

        stepper = ODEStepper(
            name='', solver=None, entity_list=[e, e1], prop_name='position',
            integrands=['u', 'v'], integrals=['x', 'y'], time_step=ts)

        stepper.setup_component()
        
        self.assertEqual(stepper.setup_done, True)
        self.assertEqual(stepper.next_step_names, ['x_next', 'y_next'])
        parr = e.get_particle_array()

        self.assertEqual(parr.properties.has_key('x_next'), True)
        self.assertEqual(parr.properties.has_key('y_next'), True)
        self.assertEqual(stepper.entity_list, [e])

    def test_compute(self):
        """
        Tests the compute function.
        """
        e = get_ode_step_data()
        e1 = EntityBase()
        ts = TimeStep(1.0)
        stepper = ODEStepper(name='', solver=None, entity_list=[e],
                             prop_name='position', integrands=['u'],
                             integrals=['x'], time_step=ts)

        stepper.py_compute()

        # setup_component must have been called once atleast.
        self.assertEqual(stepper.setup_done, True)

        parr = e.get_particle_array()
        x_next = [-2.0, 1.0, 1.0]
        self.assertEqual(check_array(x_next, parr.x_next), True)

        e = get_ode_step_data()
        stepper = ODEStepper(
            name='', solver=None, entity_list=[e, e1], prop_name='position',
            integrands=['u', 'v'], integrals=['x', 'y'], time_step=ts)
        stepper.py_compute()
        self.assertEqual(stepper.setup_done, True)
        parr = e.get_particle_array()
        y_next = [1.0, 1.0, 0.0]
        self.assertEqual(check_array(parr.x_next, x_next), True)
        self.assertEqual(check_array(parr.y_next, y_next), True)

        # step by 0.5 time step
        ts.value = 0.5
        stepper.py_compute()
        x_next = [-1.5, 0.5, 1.0]
        y_next = [1.0, 0.5, -0.5]
        self.assertEqual(check_array(parr.x_next, x_next), True)
        self.assertEqual(check_array(parr.y_next, y_next), True)

class TestIntegrator(unittest.TestCase):
    """
    Tests the Integrator class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        i = Integrator()
        
        ip = i.information[i.INTEGRATION_PROPERTIES]
        self.assertEqual(len(ip), 2)
        self.assertEqual(ip.has_key('velocity'), True)
        self.assertEqual(ip.has_key('position'), True)

        ds = i.information[i.DEFAULT_STEPPERS]
        self.assertEqual(ds.has_key('default'), True)
        self.assertEqual(ds['default'], 'euler')

    def test_add_property(self):
        """
        Tests the add_property function.
        """
        i = Integrator()
        
        prop_name = 'density'
        integrand_arrays = ['rho_rate']
        integral_arrays = ['rho']
        entity_types = [Fluid]
        stepper = {'default':'euler',
                   Fluid:'ya_stepper'} 

        i.add_property(prop_name, integrand_arrays, integral_arrays,
                       entity_types, stepper)

        ip = i.information[i.INTEGRATION_PROPERTIES]

        self.assertEqual(ip.has_key('density'), True)
        density_info = ip['density']
        self.assertEqual(density_info['integral'], ['rho'])
        self.assertEqual(density_info['integrand'], ['rho_rate'])
        self.assertEqual(density_info['entity_types'],
                         [Fluid])
        self.assertEqual(len(density_info['steppers']), 2)
        self.assertEqual(density_info['steppers']['default'],
                         'euler')
        self.assertEqual(density_info['steppers'][Fluid],
                         'ya_stepper')

    def test_add_per_step_component(self):
        """
        Tests the add_pre_step_component and add_post_step_component function.
        """
        i = Integrator()
        i.add_pre_step_component('pre_v_1', 'velocity')
        i.add_pre_step_component('pre_v_2', 'velocity')
        i.add_post_step_component('post_v_1', 'velocity')

        ip = i.information[i.INTEGRATION_PROPERTIES]
        
        vel_info = ip['velocity']
        pre_comps = vel_info['pre_step_components']
        self.assertEqual(pre_comps, ['pre_v_1', 'pre_v_2'])
        post_comps = vel_info['post_step_components']
        self.assertEqual(post_comps, ['post_v_1'])

        # add a new property and add components to it.
        prop_name = 'density'
        integrand_arrays = ['rho_rate']
        integral_arrays = ['rho']
        entity_types = [Fluid]
        stepper = {'default':'euler',
                   Fluid:'ya_stepper'} 

        i.add_property(prop_name, integrand_arrays, integral_arrays,
                       entity_types, stepper)

        i.add_pre_step_component('pre_den_1', 'density')
        den_info = ip['density']
        pre_comps = den_info['pre_step_components']
        self.assertEqual(pre_comps, ['pre_den_1'])
        self.assertEqual(den_info.get('post_step_components'), None)

    def test_add_pre_integration_component(self):
        """
        Tests the add_pre_integration_component function.
        """
        i = Integrator()

        i.add_pre_integration_component('comp1')
        i.add_pre_integration_component('comp2')
        i.add_pre_integration_component('comp0', at_tail=False)

        pic = i.information['PRE_INTEGRATION_COMPONENTS']
        self.assertEqual(pic[0], 'comp0')
        self.assertEqual(pic[1], 'comp1')
        self.assertEqual(pic[2], 'comp2')

    def test_add_post_integration_component(self):
        """
        Tests the add_post_step_component function.
        """
        i = Integrator()

        i.add_post_integration_component('c1')
        i.add_post_integration_component('c2')
        i.add_post_integration_component(comp_name='c3', at_tail=False)
        
        pic = i.information[i.POST_INTEGRATION_COMPONENTS]
        print pic
        self.assertEqual(pic[0], 'c3')
        self.assertEqual(pic[1], 'c1')
        self.assertEqual(pic[2], 'c2')

    def test_set_integration_order(self):
        """
        Tests the set_integration_order function.
        """
        i = Integrator()

        # add a new property and add components to it.
        prop_name = 'density'
        integrand_arrays = ['rho_rate']
        integral_arrays = ['rho']
        entity_types = [Fluid]
        stepper = {'default':'euler',
                   Fluid:'ya_stepper'} 

        i.add_property(prop_name, integrand_arrays, integral_arrays,
                       entity_types, stepper)

        i.set_integration_order(['density', 'velocity', 'position'])

        io = i.information[i.INTEGRATION_ORDER]
        self.assertEqual(io, ['density', 'velocity', 'position'])
        i.set_integration_order(['density'])
        self.assertEqual(io, ['density'])

    def test_get_stepper(self):
        """
        Tests the get_stepper function.
        """
        i = Integrator()
        prop_name = 'density'
        integrand_arrays = ['rho_rate']
        integral_arrays = ['rho']
        entity_types = [Fluid]
        stepper = {'default':'euler',
                   Fluid:'ya_stepper'} 
        
        i.add_property(prop_name, integrand_arrays, integral_arrays, 
                       entity_types, stepper)

        print i.information[i.INTEGRATION_PROPERTIES]

        s = i.get_stepper(Fluid, 'velocity')
        self.assertEqual(type(s), ODEStepper)
        s = i.get_stepper(Solid, 'velocity')
        self.assertEqual(type(s), ODEStepper)
        s = i.get_stepper(EntityBase, 'velocity')
        self.assertEqual(type(s), ODEStepper)
        s = i.get_stepper(EntityBase, 'position')
        self.assertEqual(type(s), ODEStepper)
        s = i.get_stepper(Fluid, 'position')
        self.assertEqual(type(s), ODEStepper)
        s = i.get_stepper(Solid, 'position')
        self.assertEqual(type(s), ODEStepper)

        s = i.get_stepper(Solid, 'density')
        self.assertEqual(type(s), ODEStepper)
        s = i.get_stepper(EntityBase, 'density')
        self.assertEqual(type(s), ODEStepper)
        s = i.get_stepper(Fluid, 'density')
        self.assertEqual(type(s), YAStepper)        

    def test_update_property_requirements(self):
        """
        Tests the update_property_requirements function.
        """
        i = Integrator()
        
        prop_name = 'density'
        integrand_arrays = ['rho_rate']
        integral_arrays = ['rho']
        entity_types = [Fluid]
        i.add_property(prop_name, integrand_arrays, integral_arrays,
                       entity_types)
        
        i.add_entity_type('velocity', EntityBase)
        i.add_entity_type('position', EntityBase)

        
        # make sure NO property information is present currently.
        self.assertEqual(i.particle_props_read, {})
        self.assertEqual(i.particle_props_write, {})
        self.assertEqual(i.particle_props_private, {})
        
        i.update_property_requirements()

        self.assertEqual(i.particle_props_read, {})
        self.assertEqual(i.particle_props_private, {})

        # make sure the write properties were added.
        wp = i.particle_props_write
        
        self.assertEqual(wp.has_key(EntityBase), True)
        self.assertEqual(wp.has_key(Fluid), True)

        f_props = wp[Fluid]
        # read each prop into a dict and confirm the necessary properties
        # are there.
        f_dict = {}
        for f_prop in f_props:
            f_dict[f_prop['name']] = f_prop['name']

        self.assertEqual(f_dict.has_key('rho'), True)
        self.assertEqual(f_dict.has_key('rho_rate'), True)

        # base entity props
        b_props = wp[EntityBase]
        b_dict = {}
        for b_prop in b_props:
            b_dict[b_prop['name']] = b_prop['name']
        self.assertEqual(b_dict.has_key('x'), True)
        self.assertEqual(b_dict.has_key('y'), True)
        self.assertEqual(b_dict.has_key('z'), True)

        self.assertEqual(b_dict.has_key('u'), True)
        self.assertEqual(b_dict.has_key('v'), True)
        self.assertEqual(b_dict.has_key('w'), True)

        self.assertEqual(b_dict.has_key('ax'), True)
        self.assertEqual(b_dict.has_key('ay'), True)
        self.assertEqual(b_dict.has_key('az'), True)

    def test_setup_component(self):
        """
        Tests the setup_component function.
        """
        cm, i, e1, e2, e3 = get_sample_integrator_setup()

        e2.set_properties_to_integrate(['position'])
        e3.set_properties_to_integrate(['density'])

        i.add_entity(e1)
        i.add_entity(e2)
        i.add_entity(e3)

        i.setup_component()

        for c in i.execute_list:
            print c.name
        
        # make sure the execute_list was setup properly.
        self.assertEqual(len(i.execute_list), 17)
        
        self.assertEqual(i.execute_list[0], cm.get_component('c1'))
        self.assertEqual(i.execute_list[1], cm.get_component('c2'))

        self.assertEqual(type(i.execute_list[2]), ArrayInitializer)
        ai = i.execute_list[2]
        self.assertEqual(ai.array_names, ['ax', 'ay', 'az'])
        self.assertEqual(ai.array_values, [0, 0, 0])
        self.assertEqual(ai.entity_list, [e1])

        self.assertEqual(i.execute_list[3], cm.get_component('c3'))
        
        s1 = i.execute_list[4]
        self.check_stepper(s1, 'velocity', ODEStepper, [e1], ['ax', 'ay', 'az'],
                           ['u', 'v', 'w'])

        self.assertEqual(i.execute_list[5], cm.get_component('c4'))
        self.assertEqual(i.execute_list[6], cm.get_component('c5'))
        
        s2 = i.execute_list[7]
        self.check_stepper(s2, 'position', ODEStepper, [e1], ['u', 'v', 'w'],
                           ['x', 'y', 'z'])
        s3 = i.execute_list[8]
        self.check_stepper(s3, 'position', ODEStepper, [e2], ['u', 'v', 'w'],
                           ['x', 'y', 'z'])
        s4 = i.execute_list[9]
        self.check_stepper(s4, 'density', YAStepper, [e1], ['rho_rate'],
                           ['rho']) 
        s5 = i.execute_list[10]
        self.check_stepper(s5, 'density', ODEStepper, [e3], ['rho_rate'],
                           ['rho']) 
     
        # now test for the copiers.
        cp1 = i.execute_list[11]
        self.check_copier(cp1, [e1], 
                         ['u_next', 'v_next', 'w_next'], 
                         ['u', 'v', 'w'])
        cp2 = i.execute_list[12]
        self.check_copier(cp2, [e1],
                          ['x_next', 'y_next', 'z_next'],
                          ['x', 'y', 'z'])
        cp3 = i.execute_list[13]
        self.check_copier(cp3, [e2],
                          ['x_next', 'y_next', 'z_next'],
                          ['x', 'y', 'z'])
        cp4 = i.execute_list[14]
        self.check_copier(cp4, [e1],
                          ['rho_next'],
                          ['rho'])
        cp5 = i.execute_list[15]
        self.check_copier(cp5, [e3],
                          ['rho_next'],
                          ['rho'])

        self.assertEqual(i.execute_list[16], cm.get_component('c6'))

    def check_copier(self, cp_obj, entity_list, from_arr, to_arr):
        self.assertEqual(cp_obj.entity_list, entity_list)
        self.assertEqual(cp_obj.from_arrays, from_arr)
        self.assertEqual(cp_obj.to_arrays, to_arr)

    def check_stepper(self, s_obj, prop_name, e_type, entity_list, integrand_arr,
                      integral_arr):
        self.assertEqual(s_obj.prop_name, prop_name)
        self.assertEqual(type(s_obj), e_type)
        self.assertEqual(s_obj.entity_list, entity_list)
        self.assertEqual(s_obj.integrand_names, integrand_arr)
        self.assertEqual(s_obj.integral_names, integral_arr)


if __name__ == '__main__':
    import logging
    logger = logging.getLogger()
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    unittest.main()

