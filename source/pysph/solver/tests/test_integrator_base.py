"""
Tests for the integrator_base module.
"""


# standard import 
import unittest

# local imports
from pysph.solver.integrator_base import Integrator


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

