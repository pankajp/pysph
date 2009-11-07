"""
Tests for the integrator_base module.
"""


# standard import 
import unittest
import numpy

# local imports
from pysph.base.particle_array import ParticleArray

from pysph.solver.entity_base import EntityBase, Fluid
from pysph.solver.entity_types import EntityTypes
from pysph.solver.integrator_base import Integrator, TimeStep, ODEStepper
from pysph.solver.solver_component import ComponentManager
from pysph.solver.dummy_components import *

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
    """

    # setup a component manager with some components.
    c = ComponentManager()

    # create a few components to be used in the integrator and add them to the
    # component manager.
    c1 = DummyComponent1('c1')
    c.add_component(c1)
    c2 = DummyComponent3('c2')
    c.add_component(c2)
    c3 = DummyComponent3('c3')
    c.add_component(c3)
    c4 = DummyComponent1('c4')
    c.add_component(c4)
    c5 = DummyComponent3('c5')
    c.add_component(c5)
    c6 = DummyComponent1('c6')
    c.add_component(c6)

    # create the required entities.
    props = {}
    standard_props={'x':{}, 'y':{}, 'z':{}, 'u':{}, 'v':{}, 'w':{}, 'ax':{},
                  'ay':{}, 'az':{}}

    props.update(c.get_particle_properties(EntityTypes.Entity_Fluid))
    # add velocity and position properties
    props.update(standard_props)
    e1 = Fluid(particle_props=props)
    e2 = Fluid(particle_props=props)

    props1 = {}
    props1.update(c.get_particle_properties(EntityTypes.Entity_Solid))
    props1.update(standard_props)
    e3= SimpleEntity(particle_props=props1)

    # now setup the integrator.
    i = Integrator()
    
    
    prop_name = 'density'
    integrand_arrays = ['rho_rate']
    integral_arrays = ['rho']
    entity_types = [EntityTypes.Entity_Fluid]
    stepper = {'default':'euler',
               EntityTypes.Entity_Fluid:'ya_stepper'} 
    
    i.add_property(prop_name, integrand_arrays, integral_arrays,
                   entity_types, stepper)

    i.add_component('velocity', 'dc1')
    i.add_component('velocity', 'dc2', True)    

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

        stepper = ODEStepper('', None, [e], 'position', ['u'], ['x'], ts)

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

        stepper = ODEStepper('', None, [e], 'position', ['u'], ['x'], ts)
        stepper.setup_component()

        self.assertEqual(stepper.setup_done, True)
        self.assertEqual(stepper.next_step_names, ['x_next'])
        parr = e.get_particle_array()

        self.assertEqual(parr.properties.has_key('x_next'), True)

        stepper = ODEStepper(
            '', None, [e, e1], 'position', ['u', 'v'], ['x', 'y'], ts
            )
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
        stepper = ODEStepper('', None, [e], 'position', ['u'], ['x'], ts)

        stepper.py_compute()

        # setup_component must have been called once atleast.
        self.assertEqual(stepper.setup_done, True)

        parr = e.get_particle_array()
        x_next = [-2.0, 1.0, 1.0]
        self.assertEqual(check_array(x_next, parr.x_next), True)

        e = get_ode_step_data()
        stepper = ODEStepper(
            '', None, [e, e1], 'position', ['u', 'v'], ['x', 'y'], ts
            )
        stepper.py_compute()
        self.assertEqual(stepper.setup_done, True)
        parr = e.get_particle_array()
        y_next = [1.0, 1.0, 0.0]
        self.assertEqual(check_array(parr.x_next, x_next), True)
        self.assertEqual(check_array(parr.y_next, y_next), True)

        # step by 0.5 time step
        ts.time_step = 0.5
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
        
        ip = i.information.get_dict(i.INTEGRATION_PROPERTIES)
        self.assertEqual(len(ip), 2)
        self.assertEqual(ip.has_key('velocity'), True)
        self.assertEqual(ip.has_key('position'), True)

        ds = i.information.get_dict(i.DEFAULT_STEPPERS)
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
        entity_types = [EntityTypes.Entity_Fluid]
        stepper = {'default':'euler',
                   EntityTypes.Entity_Fluid:'ya_stepper'} 

        i.add_property(prop_name, integrand_arrays, integral_arrays,
                       entity_types, stepper)

        ip = i.information.get_dict(i.INTEGRATION_PROPERTIES)

        self.assertEqual(ip.has_key('density'), True)
        density_info = ip['density']
        self.assertEqual(density_info['integral'], ['rho'])
        self.assertEqual(density_info['integrand'], ['rho_rate'])
        self.assertEqual(density_info['entity_types'],
                         [EntityTypes.Entity_Fluid])
        self.assertEqual(len(density_info['steppers']), 2)
        self.assertEqual(density_info['steppers']['default'],
                         'euler')
        self.assertEqual(density_info['steppers'][EntityTypes.Entity_Fluid],
                         'ya_stepper')

    def test_add_component(self):
        """
        Tests the add_component function.
        """
        i = Integrator()
        i.add_component('velocity', 'pre_v_1')
        i.add_component('velocity', 'pre_v_2')
        i.add_component('velocity', 'post_v_1', pre_step=False)

        ip = i.information.get_dict(i.INTEGRATION_PROPERTIES)
        
        vel_info = ip['velocity']
        pre_comps = vel_info['pre_step_components']
        self.assertEqual(pre_comps, ['pre_v_1', 'pre_v_2'])
        post_comps = vel_info['post_step_components']
        self.assertEqual(post_comps, ['post_v_1'])

        # add a new property and add components to it.
        prop_name = 'density'
        integrand_arrays = ['rho_rate']
        integral_arrays = ['rho']
        entity_types = [EntityTypes.Entity_Fluid]
        stepper = {'default':'euler',
                   EntityTypes.Entity_Fluid:'ya_stepper'} 

        i.add_property(prop_name, integrand_arrays, integral_arrays,
                       entity_types, stepper)

        i.add_component('density', 'pre_den_1')
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

        pic = i.information.get_list('PRE_INTEGRATION_COMPONENTS')
        self.assertEqual(pic[0], 'comp0')
        self.assertEqual(pic[1], 'comp1')
        self.assertEqual(pic[2], 'comp2')

    def test_set_integration_order(self):
        """
        Tests the set_integration_order function.
        """
        i = Integrator()

        # add a new property and add components to it.
        prop_name = 'density'
        integrand_arrays = ['rho_rate']
        integral_arrays = ['rho']
        entity_types = [EntityTypes.Entity_Fluid]
        stepper = {'default':'euler',
                   EntityTypes.Entity_Fluid:'ya_stepper'} 

        i.add_property(prop_name, integrand_arrays, integral_arrays,
                       entity_types, stepper)

        i.set_integration_order(['density', 'velocity', 'position'])

        io = i.information.get_list(i.INTEGRATION_ORDER)
        self.assertEqual(io, ['density', 'velocity', 'position'])
        i.set_integration_order(['density'])
        self.assertEqual(io, ['density'])

    def test_get_stepper(self):
        """
        Tests the get_stepper function.
        """
        i = get_sample_integrator_setup()
        pass

if __name__ == '__main__':
    import logging
    logger = logging.getLogger()
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    unittest.main()

