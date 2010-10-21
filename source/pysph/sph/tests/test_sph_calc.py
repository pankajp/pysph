"""
Tests for the sph_calc module.
"""
# standard imports
import unittest, numpy, pylab

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
    that that under the kernel support radius of 2, all particles will 
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
        pa.set(**{'type':type})

        self.kernel = kernel = base.CubicSplineKernel(dim=2)

        self.particles = particles = base.Particles(pa, kernel.radius())

        self.func = func = sph.GravityForce(pa, gx=1, gy=2, gz=3)
        
        self.setup()

    def setup(self):
        pass


class TestSPHBase(SPHTestCase):
    """ Test the constructor and default behavior """
    
    def setup(self):
        """ Construct an SPHBase instance with default properties """
        self.sph = sph.SPHBase(particles=self.particles, kernel=self.kernel,
                               func=self.func, updates=['x','y','z'],
                               from_types=[Fluid,Solid], on_types=[],
                               integrates=True)

    def test_constructor(self):
        """ Test the construction of the calc object """
        
        calc = self.sph

        #Check for the type array
        self.assertEqual(check_array(calc.type_arr, self.pa.type), True)

        #Check the update arrays
        updates = calc.updates

        for i, prop in enumerate(calc.updates):
            update_array = calc.update_arrays[i]            
            self.assertEqual(check_array(update_array,
                                         self.pa.get(updates[i])),True)

        self.assertEqual(calc.integrates, True)

    def test_integrating_sph(self):
        """ Call the sph function for this calc.

        Expected Behavior:
        -------------------
        By default, the calc is integrating. Since the on_types
        was an empty list. This should set the output arrays to zeros for 
        all partilces.

        """

        calc = self.sph
        pa = calc.source

        np = pa.get_number_of_particles()
        
        #Set the temporary arrays to some non zeros
        tmpx = numpy.zeros(np, float)
        tmpy = numpy.zeros(np, float)
        tmpz = numpy.zeros(np, float)
        zeros = numpy.zeros(np, float)

        pa.set(**{'tmpx':tmpz, 'tmpy':tmpy, 'tmpz':tmpz})

        #call sph with no arguments
        calc.sph()

        tmpx, tmpy, tmpz = pa.get('tmpx','tmpy','tmpz')

        self.assertEqual(check_array(tmpx, zeros), True)
        self.assertEqual(check_array(tmpy, zeros), True)
        self.assertEqual(check_array(tmpz, zeros), True)

    def test_non_integrating_sph(self):
        """ Test for a non integrating calc.

        Expected Behavior:
        ------------------
        If the calc is set to non integrating, then a call to sph
        should set the output array to the update variable for those 
        particles not in the on_types.
        
        The output arrays should be the position arrays.

        """
        calc = self.sph
        pa = calc.source

        np = pa.get_number_of_particles()
        
        #Set the calc to non integrating
        calc.integrates = False
        calc.sph()

        tmpx, tmpy, tmpz = pa.get('tmpx','tmpy','tmpz')
        x,y,z = pa.get('x','y','z')

        self.assertEqual(check_array(tmpx, x), True)
        self.assertEqual(check_array(tmpy, y), True)
        self.assertEqual(check_array(tmpz, z), True)


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
        self.sph = sph.SPHEquation(particles=self.particles, kernel=self.kernel,
                                   func=self.func, updates=['x','y','z'],
                                   from_types=[Fluid,Solid], on_types=[Fluid],
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
        pa = calc.source
        np = pa.get_number_of_particles()


        #Set the on types to Fluid
        calc.on_types = [Fluid]

        calc.sph()

        tmpx, tmpy, tmpz, type = pa.get('tmpx','tmpy','tmpz', 'type')
        
        for i in range(np):
            if type[i] == Fluid:
                self.assertAlmostEqual(tmpx[i], 1, 10)
                self.assertAlmostEqual(tmpy[i], 2, 10)
                self.assertAlmostEqual(tmpz[i], 3, 10)
            else:
                self.assertAlmostEqual(tmpx[i], 0, 10)
                self.assertAlmostEqual(tmpy[i], 0, 10)
                self.assertAlmostEqual(tmpz[i], 0, 10)

        
        #Set the on_types to Solid

        calc.on_types = [Solid]
        calc.sph()
                
        tmpx, tmpy, tmpz, type = pa.get('tmpx','tmpy','tmpz', 'type')

        for i in range(np):
            if type[i] == Fluid:
                self.assertAlmostEqual(tmpx[i], 0, 10)
                self.assertAlmostEqual(tmpy[i], 0, 10)
                self.assertAlmostEqual(tmpz[i], 0, 10)
            else:
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
        pa = calc.source
        np = pa.get_number_of_particles()

        calc.integrates = False
        
        #Set the on types to Fluid
        calc.on_types = [Fluid]
        
        calc.sph()
        tmpx, tmpy, tmpz, type = pa.get('tmpx','tmpy','tmpz', 'type')
        x, y, z = pa.get('x','y','z')

        for i in range(np):
            if type[i] == Fluid:
                self.assertAlmostEqual(tmpx[i], 1, 10)
                self.assertAlmostEqual(tmpy[i], 2, 10)
                self.assertAlmostEqual(tmpz[i], 3, 10)
        
            else:
                self.assertAlmostEqual(tmpx[i], x[i], 10)
                self.assertAlmostEqual(tmpy[i], y[i], 10)
                self.assertAlmostEqual(tmpz[i], z[i], 10)
        

        #Set the on types to Solid
        calc.on_types = [Solid]
        
        calc.sph()
        tmpx, tmpy, tmpz, type = pa.get('tmpx','tmpy','tmpz', 'type')
        x, y, z = pa.get('x','y','z')

        for i in range(np):
            if type[i] == Solid:
                self.assertAlmostEqual(tmpx[i], 1, 10)
                self.assertAlmostEqual(tmpy[i], 2, 10)
                self.assertAlmostEqual(tmpz[i], 3, 10)
        
            else:
                self.assertAlmostEqual(tmpx[i], x[i], 10)
                self.assertAlmostEqual(tmpy[i], y[i], 10)
                self.assertAlmostEqual(tmpz[i], z[i], 10)

###############################################################################


class TestSPHEquation(SPHTestCase):
    """ Tests for SPHCalc 

    Setup:
    ------
    The calc is given the count neighbors function which simply counts the 
    number of neighbors of a particle. The number of neighbors is tested
    for the particle with index `0` residing at the origin.
    
    """

    def setup(self):
        self.func = func = sph.CountNeighbors(self.pa)
        self.particles.update()
        
        self.sph = sph.SPHCalc(particles=self.particles, kernel=self.kernel,
                               func=self.func, updates=['x','y','z'],
                               from_types=[Fluid,Solid], on_types=[Fluid],
                               integrates=True)

        type = self.particles.pa.get('type')

        self.nfluids = len(pylab.find(type==Fluid))
        self.nsolids = len(pylab.find(type==Solid))

    def test_fluid_neighbors(self):
        calc = self.sph
        pa = calc.source
        
        calc.on_types = [Fluid, Solid]
        calc.from_types = [Fluid]

        calc.sph()
        
        nbrs = pa.get('tmpx')
        nbrs0 = nbrs[0]

        self.assertEqual(nbrs0, self.nfluids)

    def test_solid_neighbors(self):
        calc = self.sph
        pa = calc.source
        
        calc.on_types = [Fluid, Solid]
        calc.from_types = [Solid]

        calc.sph()
        
        nbrs = pa.get('tmpx')
        nbrs0 = nbrs[0]

        self.assertEqual(nbrs0, self.nsolids)

    def test_all_neighbors(self):
        calc = self.sph
        pa = calc.source
        
        calc.on_types = [Fluid, Solid]
        calc.from_types = [Solid, Fluid]

        calc.sph()
        
        nbrs = pa.get('tmpx')
        nbrs0 = nbrs[0]

        self.assertEqual(nbrs0, self.nsolids+self.nfluids)

##############################################################################
        
if __name__ == '__main__':
    unittest.main()
