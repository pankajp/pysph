"""
Tests for classes in the array_copier module.
"""

# standard imports
import unittest
import numpy

# local import
from pysph.solver.array_copier import ArrayCopier
from pysph.solver.entity_base import EntityBase, Fluid

def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-16."""
    return numpy.allclose(x, y, atol=1e-16, rtol=0)

################################################################################
# `TestArrayCopier` class.
################################################################################
class TestArrayCopier(unittest.TestCase):
    """
    Tests for the ArrayCopier class.
    """
    def test_constructor(self):
        """
        Tests for the constructor.
        """
        ac = ArrayCopier()
        self.assertEqual(ac.name, '')
        self.assertEqual(ac.cm, None)
        self.assertEqual(ac.entity_list, [])
        self.assertEqual(ac.from_arrays, [])
        self.assertEqual(ac.to_arrays, [])

        e = EntityBase()
        ac = ArrayCopier(name='ac1', cm=None, entity_list=[e], from_arrays=['a'],
                       to_arrays=['b'])
        self.assertEqual(ac.name, 'ac1')
        self.assertEqual(ac.cm, None)
        self.assertEqual(ac.entity_list, [e])
        self.assertEqual(ac.from_arrays, ['a'])
        self.assertEqual(ac.to_arrays, ['b'])
    
    def test_add_entity(self):
        """
        Tests the add_entity function.
        """
        ac = ArrayCopier()
        e1 = EntityBase()
        e2 = EntityBase()

        ac.add_entity(e1)
        ac.add_entity(e2)

        self.assertEqual(ac.entity_list, [e1, e2])

    def test_add_array_pair(self):
        """
        Tests the add_array_pair function.
        """
        ac = ArrayCopier()
        ac.add_array_pair('a', 'b')
        ac.add_array_pair('b', 'c')
        ac.add_array_pair('a', 'b')
        
        self.assertEqual(ac.from_arrays, ['a', 'b', 'a'])
        self.assertEqual(ac.to_arrays, ['b', 'c', 'b'])

    def test_setup_component(self):
        """
        Tests the setup_component function.
        """
        ac = ArrayCopier()
        ac.add_entity(EntityBase())
        ac.add_entity(EntityBase())
        ac.add_entity(EntityBase())

        self.assertEqual(ac.setup_done, False)
        ac.setup_component()
        self.assertEqual(ac.setup_done, True)
        self.assertEqual(ac.entity_list, [])

        f1 = Fluid(particle_props={'a':{'data':[1., 2., 3, 4]},
                                   'b':{'data':[4., 4., 3, 3]},
                                   'c':{'default':-1.}})
        ac.add_entity(f1)
        self.assertEqual(ac.setup_done, False)
        ac.setup_component()
        self.assertEqual(ac.entity_list, [f1])
        ac.add_array_pair('a', 'c')
        self.assertEqual(ac.setup_done, False)
        ac.setup_component()
        self.assertEqual(ac.setup_done, True)

        ac.add_array_pair('d', 'c')
        self.assertRaises(AttributeError, ac.setup_component)

    def test_compute(self):
        """
        Tests the compute function.
        """
        ac = ArrayCopier()
        f1 = Fluid(particle_props={'a':{'data':[1., 2., 3, 4]},
                                   'b':{'data':[4., 4., 3, 3]},
                                   'c':{'default':-1.}})

        f2 = Fluid(particle_props={'a':{'data':[1, 2]},
                                   'b':{'data':[4, 4]}})

        ac.add_entity(f1)
        ac.add_entity(f2)
        ac.add_entity(EntityBase())

        ac.add_array_pair('a', 'b')
        ac.py_compute()

        self.assertEqual(ac.entity_list, [f1, f2])

        parr = f1.get_particle_array()
        self.assertEqual(check_array(parr.a, parr.b), True)
        parr = f2.get_particle_array()
        self.assertEqual(check_array(parr.a, parr.b), True)

        ac.add_array_pair('c', 'd')
        self.assertRaises(AttributeError, ac.setup_component)

if __name__ == '__main__':
    unittest.main()    
