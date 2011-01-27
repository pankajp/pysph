"""
Tests for the sph_calc module.
"""
# standard imports
import unittest, numpy

import pysph.base.api as base
import pysph.sph.api as sph

Fluid = base.ParticleType.Fluid
Solid = base.ParticleType.Solid

def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-16."""
    return numpy.allclose(x, y, atol=1e-16, rtol=0)

class SPHTestCase(unittest.TestCase):
    """ Default test case for the SPHBase Class.

    Setup:
    ------
    N+1 particles are setup on a circle of rad 0.1, with alternating tags.
    The particle with index 0 is at the origin and is the query particle.

    The smoothing length of the query (all particles) is 0.1 which implies 
    that that under the kernel support radius of 0.2, all particles will 
    be neighbors.

    Refer to the figures 'test_particles.pdf' or 'test_particles.png' for
    the setup.

    """
    
    def setUp(self):

        self.theta = theta = numpy.arange(0, 2*numpy.pi, numpy.pi/100.)
        self.n = n = len(theta)
        
        self.x = x = numpy.zeros(n + 1, float)
        self.y = y = numpy.zeros(n + 1, float)
        self.h = h = numpy.ones_like(x) * 0.1
        
        self.x[1:] = 0.1*numpy.cos(theta)
        self.y[1:] = 0.1*numpy.sin(theta)
        
        self.type = type = numpy.ones_like(x)
        self.type[::2] = base.ParticleType.Fluid
        
        self.pa = pa = base.get_particle_array(x=x, y=y, h=h)

        pa.add_property({'name':'type','data':type})

        self.kernel = kernel = base.CubicSplineKernel(dim=2)

        self.particles = particles = base.Particles([pa], kernel.radius())
        self.particles.kernel = self.kernel

        self.func = func = sph.GravityForce(gx=1, gy=2, gz=3).get_func(pa,pa)
        
        self.setup()

    def setup(self):
        pass


class TestSPHBase(SPHTestCase):
    """ Test the constructor and default behavior """
    
    def setup(self):
        """ Construct an SPHBase instance with default properties """
        self.sph = sph.SPHBase(particles=self.particles, sources=[self.pa],
                               dest=self.pa, kernel=self.kernel,
                               funcs=[self.func], updates=['x','y','z'],
                               integrates=True)

    def test_constructor(self):
        """ Test the construction of the calc object """
        
        calc = self.sph

        #Check the update arrays
        updates = calc.updates
        for prop in calc.updates:
            update_array = calc.dest.get(prop)
            self.assertEqual(check_array(update_array, self.pa.get(prop)),True)

        self.assertEqual(calc.integrates, True)


    def test_sph(self):
        """ Set the on types to Fluid and assert that an error is raises 
        
        Expected Behavior:
        ------------------
        SPHBase should not call the eval function. It must raise an error.
        
        """
        calc = self.sph
        calc.on_types = [Fluid]

        self.assertRaises(NotImplementedError, calc.sph)

###############################################################################

class TestSPHEquation(SPHTestCase):
    """ Tests for integrating and non integrating versions of SPHEquation """

    def setup(self):
        self.sph = sph.SPHEquation(particles=self.particles, sources=[self.pa],
                               dest=self.pa, kernel=self.kernel,
                               funcs=[self.func], updates=['x','y','z'],
                               integrates=True)


    def test_integrating_sph(self):
        """ Test for the integrating version of SPHEquation

        Expected Behavior:
        ------------------

        When the on_types is set to Fluid, a call to sph should update
        the output arrays with the function provided and set the other 
        values to zero for the non solids
        
        """

        calc = self.sph
        pa = calc.sources[0]
        np = pa.get_number_of_particles()

        calc.sph()

        tmpx, tmpy, tmpz, type = pa.get('tmpx','tmpy','tmpz', 'type')
        
        for i in range(np):
            self.assertAlmostEqual(tmpx[i], 1, 10)
            self.assertAlmostEqual(tmpy[i], 2, 10)
            self.assertAlmostEqual(tmpz[i], 3, 10)

        
        #Set the on_types to Solid

        calc.on_types = [Solid]
        calc.sph()
                
        tmpx, tmpy, tmpz, type = pa.get('tmpx','tmpy','tmpz', 'type')

        for i in range(np):
            self.assertAlmostEqual(tmpx[i], 1, 10)
            self.assertAlmostEqual(tmpy[i], 2, 10)
            self.assertAlmostEqual(tmpz[i], 3, 10)

    def test_non_integrating_sph(self):
        """ Tests for the non integrating version of SPHEquation 

        Expected Behavior:
        ------------------
        The on_types should get a value from the function and the other
        types should have the value set from the update property
        
        """
        calc = self.sph
        pa = calc.sources[0]
        np = pa.get_number_of_particles()

        calc.integrates = False
        
        #Set the on types to Fluid
        calc.on_types = [Fluid]
        
        calc.sph()
        tmpx, tmpy, tmpz, type = pa.get('tmpx','tmpy','tmpz', 'type')
        x, y, z = pa.get('x','y','z')

        for i in range(np):
            self.assertAlmostEqual(tmpx[i], 1, 10)
            self.assertAlmostEqual(tmpy[i], 2, 10)
            self.assertAlmostEqual(tmpz[i], 3, 10)
        

        #Set the on types to Solid
        calc.on_types = [Solid]
        
        calc.sph()
        tmpx, tmpy, tmpz, type = pa.get('tmpx','tmpy','tmpz', 'type')
        x, y, z = pa.get('x','y','z')

        for i in range(np):
            self.assertAlmostEqual(tmpx[i], 1, 10)
            self.assertAlmostEqual(tmpy[i], 2, 10)
            self.assertAlmostEqual(tmpz[i], 3, 10)


###############################################################################


class TestSPHCalc(SPHTestCase):
    """ Tests for SPHCalc 

    Setup:
    ------
    The calc is given the count neighbors function which simply counts the 
    number of neighbors of a particle. The number of neighbors is tested
    for the particle with index `0` residing at the origin.
    
    """

    def setup(self):
        self.func = func = sph.NeighborCount().get_func(self.pa, self.pa)
        self.particles.update()
        
        self.sph = sph.SPHCalc(particles=self.particles, sources=[self.pa],
                               dest=self.pa, kernel=self.kernel,
                               funcs=[self.func], updates=['x','y','z'],
                               integrates=True)

    def test_all_neighbors(self):
        calc = self.sph
        pa = calc.sources[0]
        
        calc.on_types = [Fluid, Solid]
        calc.from_types = [Solid, Fluid]

        calc.sph()

        nbrs = pa.get('tmpx')

        nbrs0 = nbrs[0]

        self.assertEqual(nbrs0, pa.get_number_of_particles())

##############################################################################
        
if __name__ == '__main__':
    unittest.main()
