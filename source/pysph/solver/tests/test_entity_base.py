"""
Tests for the entity_base module.
"""

# standard imports
import unittest

# local imports
from pysph.solver import entity_base


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

        self.assertEqual(e.information.get_number_of_keys(), 0)
        self.assertEqual(len(e.properties.keys()), 2)
        self.assertEqual(e.properties['a'], 10.)
        self.assertEqual(e.properties['b'], 3.)


if __name__ == '__main__':
    unittest.main()
