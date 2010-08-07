"""
Tests for the particle_generator class.
"""

# standard imports
import unittest


# local imports
from pysph.base.particle_array import ParticleArray
from pysph.solver.particle_generator import ParticleGenerator
from pysph.solver.particle_generator import DensityComputationMode as DCM
from pysph.solver.particle_generator import MassComputationMode as MCM
from pysph.base.kernels import CubicSplineKernel


class DummyGenerator(ParticleGenerator):
    """
    """
    def __init__(self):
        ParticleGenerator.__init__(self)
        self.num = 1

    def num_output_arrays(self):
        print 'returning %d'%(self.num)
        return self.num

################################################################################
# `TestParticleGenerator` class.
################################################################################
class TestParticleGenerator(unittest.TestCase):
    """
    Tests the ParticleGenerator class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        pg = ParticleGenerator()

        self.assertEqual(pg.output_particle_arrays, [])
        self.assertEqual(pg.density_computation_mode, DCM.Set_Constant)
        self.assertEqual(pg.particle_density, 1000.0)
        self.assertEqual(pg.mass_computation_mode, MCM.Compute_From_Density)
        self.assertEqual(pg.particle_mass, -1.0)
        self.assertEqual(pg.particle_h, 0.1)
    
    def test_get_particles(self):
        """
        Tests the get_particles function.
        """
        pg = ParticleGenerator()
        self.assertEqual(pg.validate_setup(), False)

        pg.kernel = CubicSplineKernel()
        self.assertRaises(NotImplementedError, pg.get_particles)
        self.assertRaises(NotImplementedError, pg.get_coords)
        self.assertRaises(NotImplementedError, pg.generate_func)

    def test_setup_outputs(self):
        """
        """
        dg = DummyGenerator()
        dg._setup_outputs()

        self.assertEqual(len(dg.output_particle_arrays), 1)

        oa = dg.output_particle_arrays[0]
        self.assertEqual(oa.properties.has_key('x'), True)
        self.assertEqual(oa.properties.has_key('y'), True)
        self.assertEqual(oa.properties.has_key('z'), True)
        self.assertEqual(oa.properties.has_key('m'), True)
        self.assertEqual(oa.properties.has_key('rho'), True)
        self.assertEqual(oa.properties.has_key('h'), True)

        p = ParticleArray(x={'data':[1, 2, 3]})
        dg = DummyGenerator()
        dg.output_particle_arrays.append(p)
        dg._setup_outputs()

        self.assertEqual(len(dg.output_particle_arrays), 1)
        self.assertEqual(dg.output_particle_arrays[0], p)

        oa = p
        self.assertEqual(oa.properties.has_key('x'), True)
        self.assertEqual(oa.properties.has_key('y'), True)
        self.assertEqual(oa.properties.has_key('z'), True)
        self.assertEqual(oa.properties.has_key('m'), True)
        self.assertEqual(oa.properties.has_key('rho'), True)
        self.assertEqual(oa.properties.has_key('h'), True)
        
if __name__ == '__main__':
    unittest.main()
