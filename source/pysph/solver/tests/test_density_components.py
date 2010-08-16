"""
Tests for the density_components module.
"""


# standard imports
import unittest


# local imports
from pysph.solver.density_components import SPHDensityComponent
from pysph.solver.solid import Solid
from pysph.solver.fluid import Fluid

def get_sample_data():
    """
    """
    pass

class TestSPHDensityComponent(unittest.TestCase):
    """
    Tests the SPHDensityComponent class.
    """
    def test_constructor(self):
        """
        """
        c = SPHDensityComponent()

        self.assertEqual(c.sph_calcs, [])
        self.assertEqual(c.dest_list, [])

        self.assertEqual(c.kernel, None)
        self.assertEqual(c.name, '')
        self.assertEqual(c.source_types, [Fluid])
        self.assertEqual(c.dest_types, [Fluid])
        

if __name__ == '__main__':
    unittest.main()
