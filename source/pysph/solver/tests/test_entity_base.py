"""
Tests for the entity_base module.
"""

# standard imports
import unittest

# local imports
from pysph.solver.entity_base import EntityBase
from pysph.solver.fluid import Fluid
from pysph.solver.solid import Solid

################################################################################
# `TestEntityBase` class.
################################################################################

class TestEntityBase(unittest.TestCase):
    """
    Tests for the EntityBase class.
    """
    def test_constructor(self):
        """
        Tests for the constructor.
        """

        e = EntityBase(properties={'a':10., 'b':3.})
        self.assertEqual(len(e.information), 1)
        self.assertEqual(len(e.properties.keys()), 2)
        self.assertEqual(e.properties['a'], 10.)
        self.assertEqual(e.properties['b'], 3.)

    def test_is_type_included(self):
        """
        Tests the is_type_included function.
        """
        e = EntityBase()
        tlist = [Fluid]
        
        self.assertEqual(e.is_type_included(tlist), False)
        
        tlist.append(EntityBase)
        self.assertEqual(e.is_type_included(tlist), True)

        e = Fluid()
        tlist = [EntityBase]
        self.assertEqual(e.is_type_included(tlist), True)

        tlist = [Fluid]
        self.assertEqual(e.is_type_included(tlist), True)
        
        tlist = [Solid]
        self.assertEqual(e.is_type_included(tlist), False)

if __name__ == '__main__':
    unittest.main()
