"""
Tests for the base module.
"""

# standard modules
import unittest

# local imports
from pysph.solver.base import *



class TestBase(unittest.TestCase):
    """
    Tests for the Base class.
    """
    def test_constructor(self):
        """
        Tests for the constructor.
        """
        b = Base()
        self.assertEqual(b.information.get_number_of_keys(), 0)


if __name__ == '__main__':
    unittest.main()
