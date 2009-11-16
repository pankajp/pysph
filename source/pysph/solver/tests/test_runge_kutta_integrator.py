"""
Tests for classes in the runge_kutta_integrator module.
"""

# standard imports
import unittest
import logging
logger = logging.getLogger()


# local imports
from pysph.solver.runge_kutta_integrator import *
from pysph.solver.tests.test_integrator_base import get_sample_integrator_setup

################################################################################
# `TestRK2SecondStep` class.
################################################################################
class TestRK2SecondStep(unittest.TestCase):
    """
    """
    pass
################################################################################
# `TestRK2Integrator` class.
################################################################################
class TestRK2Integrator(unittest.TestCase):
    """
    Tests for the class RK2Integrator.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        rk2i = RK2Integrator()

        

        

if __name__ == '__main__':
    unittest.main()
