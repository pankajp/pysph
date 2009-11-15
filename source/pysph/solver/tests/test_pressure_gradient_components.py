"""
Tests for the pressure_gradient_components module.
"""

# standard imports
import unittest

# local imports
from pysph.solver.pressure_gradient_components import \
    SPHSymmetricPressureGradientComponent as SSPgComponent



class TestSSPgComponent(unittest.TestCase):
    """
    Tests the SPHSymmetricPressureGradientComponent class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        c = SSPgComponent()

        pass


if __name__ == '__main__':
    unittest.main()
    
