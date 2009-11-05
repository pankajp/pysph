"""
Tests for the integrator_base module.
"""


# standard import 
import unittest
import numpy

# local imports
from pysph.base.particle_array import ParticleArray

from pysph.solver.entity_base import EntityBase
from pysph.solver.integrator_base import Integrator, TimeStep, ODESteper

def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-16."""
    return numpy.allclose(x, y, atol=1e-16, rtol=0)

class SimpleEntity(EntityBase):
    """
    Simple entity class for test purposes.
    """
    def __init__(self, name=''):
        """
        """
        self.parr = None

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

class TestODEStepper(unittest.TestCase):
    """
    Tests the ODESteper class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        e = get_ode_step_data()
        ts = TimeStep(1.0)

        stepper = ODESteper('', None, [e], 'position', ['u'], ['x'], ts)

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

        stepper = ODESteper('', None, [e], 'position', ['u'], ['x'], ts)
        stepper.setup_component()

        self.assertEqual(stepper.setup_done, True)
        self.assertEqual(stepper.next_step_names, ['x_next'])
        parr = e.get_particle_array()

        self.assertEqual(parr.properties.has_key('x_next'), True)

        stepper = ODESteper(
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
        stepper = ODESteper('', None, [e], 'position', ['u'], ['x'], ts)

        stepper.py_compute()

        # setup_component must have been called once atleast.
        self.assertEqual(stepper.setup_done, True)

        parr = e.get_particle_array()
        x_next = [-2.0, 1.0, 1.0]
        self.assertEqual(check_array(x_next, parr.x_next), True)

        e = get_ode_step_data()
        stepper = ODESteper(
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

        print i.information.get_dict(i.INTEGRATION_PROPERTIES)


if __name__ == '__main__':
    import logging
    logger = logging.getLogger()
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    unittest.main()

