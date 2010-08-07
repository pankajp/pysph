"""
Tests for the entity_base module.
"""

# standard imports
import unittest

# local imports
from pysph.solver import entity_base
from pysph.solver.fluid import Fluid
from pysph.solver.entity_types import *

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

        e = entity_base.EntityBase(properties={'a':10., 'b':3.})
        self.assertEqual(e.information.get_number_of_keys(), 1)
        self.assertEqual(len(e.properties.keys()), 2)
        self.assertEqual(e.properties['a'], 10.)
        self.assertEqual(e.properties['b'], 3.)

    def test_is_type_included(self):
        """
        Tests the is_type_included function.
        """
        e = entity_base.EntityBase()
        tlist = [EntityTypes.Entity_Fluid]
        
        self.assertEqual(e.is_type_included(tlist), False)
        
        tlist.append(EntityTypes.Entity_Base)
        self.assertEqual(e.is_type_included(tlist), True)

        e = Fluid()
        tlist = [EntityTypes.Entity_Base]
        self.assertEqual(e.is_type_included(tlist), True)

        tlist = [EntityTypes.Entity_Fluid]
        self.assertEqual(e.is_type_included(tlist), True)
        
        tlist = [EntityTypes.Entity_Solid]
        self.assertEqual(e.is_type_included(tlist), False)

if __name__ == '__main__':
    unittest.main()
