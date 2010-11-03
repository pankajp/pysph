""" Tests for the integrator """
import numpy

#pysph imports 
import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph

from pysph.solver.integrator import Integrator

import unittest

class IntegratorTestCase(unittest.TestCase):
    """ Tests for the Integrator base class 

    Setup a default simulation in 2D. The setup consists of four particles
    on a circle of radius 2/pi. The particles are constrained to move
    along the circle with forcing functions defined in 
    pysph/sph/funcs/external_forces.pyx 

    With the choice of radius and unit magnitude of velocity, the particles
    move by pi/2 radians in 1 second. 

    Other properties may further be integrated to test for the multiple 
    property integration provided by the integrator.

    The tests cover initialization and the proper setting up of the internal
    data for the particle arrays and integrator.
    
    """

    def runTest(self):
        pass
    
    def setUp(self):
        """
        Setup a default simulation in 2D. The setup consists of four particles
        on a circle of radius 2/pi. The particles are constrained to move
        along the circle with forcing functions defined in 
        pysph/sph/funcs/external_forces.pyx 
        
        With the choice of radius and unit magnitude of velocity, the particles
        move by pi/2 radians in 1 second. 

        """
        self.r = r = 2./numpy.pi
        self.x = x = numpy.array([1.0, 0.0, -1.0, 0.0])
        self.y = y = numpy.array([0.0, 1.0, 0.0, -1.0])
        
        x *= r
        y *= r

        p = numpy.zeros_like(x)
        e = numpy.zeros_like(x)

        self.pa = pa = base.get_particle_array(x=x,y=y,p=p,e=e)

        self.particles = particles = base.Particles(arrays=[pa])
        self.kernel = base.CubicSplineKernel(dim=2)
        
        circlex = solver.SPHSimpleODE(sph.MoveCircleX(), 
                                      from_types=[0], on_types=[0],
                                      updates=['x','y'], id = 'circlex')

        circley = solver.SPHSimpleODE(sph.MoveCircleY(), 
                                      from_types=[0], on_types=[0],
                                      updates=['x','y'], id = 'circley')
        
        self.calcx = circlex.get_calcs(particles, self.kernel)
        self.calcy = circley.get_calcs(particles, self.kernel)

        self.calcs = []
        self.calcs.extend(self.calcx)
        self.calcs.extend(self.calcy)

        self.integrator = Integrator(particles=particles, 
                                     calcs=self.calcs)

        self.setup()

    def setup(self):
        pass

    def test_constructor(self):
        """ Some constructor tests """
        self.assertEqual(self.integrator.nsteps, 1)
        self.assertEqual(self.integrator.calling_sequence, [])

    def _test_setup_integrator(self):
        """ Test the setup of the integrator

        Expected Behavior:
        ------------------
        The setup integrator function is the most important as it 
        defines the calling sequence for the sph operations, sets the
        initial arrays and the `k` arrays for multi step integrators.

        Since this integrator has two calcs, each updating the same two
        properties 'x', and 'y', the integrator should create the following:

        calling sequence: [['x_00','y_01'], ['x_10','y_11']]
        initial_arrays:   x_0 and y_0 in the particle array
        step_properties: x_00, y_01, x_10, y_11 in the particle array
        
        the `k` arrays:
        ---------------
        for the default integrator, the nsteps is 1 thus the lenght of the 
        `k` list should be 1.

        This dictionary is indexed on the calc number and then by the update
        array name. The value upon initialization should be None

        """

        integrator = self.integrator
        
        #Setup the integrator
        
        integrator.setup_integrator()

        #Check for the calling sequence

        calling_sequence = self.integrator.calling_sequence
        self.assertEqual(calling_sequence, [['x_00','y_01'], ['x_10','y_11']])
    
        #Check fot the initial and step arrays

        pa = integrator.pa
        props = pa.properties.keys()

        self.assertEqual('x_00' in props, True)
        self.assertEqual('x_10' in props, True)
        self.assertEqual('y_01' in props, True)
        self.assertEqual('y_11' in props, True)
        self.assertEqual('x_0' in props, True)
        self.assertEqual('y_0' in props, True)

    def print_pos(self, scheme, old, new):
        """ Pretty printing of the positions """
        print scheme
        for i in range(4):
            print 'Original: %s, New: %s '%(old[i],new[i])            

##############################################################################

class TestEulerIntegrator(IntegratorTestCase):
    """ Test for the Euler Integrator

    For the test, the particles (defined in the setUp of the base class)
    are constrained to move on a circle of radius 2./pi. 

    Four particles start the motion from the points ENWS and after one 
    second, the positions should be NWSE respectively.

    """
    def setup(self):
        self.integrator = solver.EulerIntegrator(particles=self.particles, 
                                                 calcs = self.calcs)

        
    def test_motion(self):
        """ Perform the integration of the particle positons 

        The scheme is the Euler integrator which is first order accurate 
        in time. The time step used for the integration is 1e-3 and thus
        we expect the positions of the particles to be exact to within 
        two decimal places.       

        """

        #setup the integrator

        self.integrator.setup_integrator()

        #set the time constants

        t = 0.0; tf = 1.0; dt = 1e-3
        
        integrator = self.integrator
        particles = integrator.particles
        pa = particles.arrays[0]

        original_pos = [(pa.x[i] ,pa.y[i]) for i in range(len(pa.x))]

        exact = (0.0, self.r), (-self.r, 0.0), (0.0, -self.r), (self.r, 0.0)

        while t <= tf:
            t += dt
            particles.update()
            integrator.integrate(dt)
            
        new_pos = [(pa.x[i] ,pa.y[i]) for i in range(len(pa.x))]
        self.print_pos('Euler Integration', original_pos, new_pos)

        for i in range(4):
            self.assertAlmostEqual(new_pos[i][0], exact[i][0], 2)
            self.assertAlmostEqual(new_pos[i][1], exact[i][1], 2)

        
##############################################################################

class TestRK2Integrator(IntegratorTestCase):
    """ Test for the Euler Integrator

    For the test, the particles (defined in the setUp of the base class)
    are constrained to move on a circle of radius 2./pi. 

    Four particles start the motion from the points ENWS and after one 
    second, the positions should be NWSE respectively.

    """
    def setup(self):
        self.integrator = solver.RK2Integrator(particles=self.particles, 
                                               calcs = self.calcs)

    def test_constructor(self):
        """ Some constructor tests """
        self.assertEqual(self.integrator.nsteps, 2)
        self.assertEqual(self.integrator.calling_sequence, [])
        
    def test_motion(self):
        """ Perform the integration of the particle positons 

        The scheme is the RK2 integrator which is first order accurate 
        in time. The time step used for the integration is 1e-3 and thus
        we expect the positions of the particles to be exact to within 
        four decimal places.       

        """

        #setup the integrator

        self.integrator.setup_integrator()

        #set the time constants

        t = 0.0; tf = 1.0; dt = 1e-3
        
        integrator = self.integrator
        particles = integrator.particles
        pa = particles.arrays[0]

        original_pos = [(pa.x[i], pa.y[i]) for i in range(len(pa.x))]
        exact = (0.0, self.r), (-self.r, 0.0), (0.0, -self.r), (self.r, 0.0)

        while t <= tf:
            t += dt
            particles.update()
            integrator.integrate(dt)

        new_pos = [(pa.x[i] ,pa.y[i]) for i in range(len(pa.x))]
        self.print_pos('RK2 Integration', original_pos, new_pos)

        for i in range(4):
            self.assertAlmostEqual(new_pos[i][0], exact[i][0], 6)
            self.assertAlmostEqual(new_pos[i][1], exact[i][1], 6)            

        
##############################################################################

class TestRK4Integrator(IntegratorTestCase):
    """ Test for the Euler Integrator

    For the test, the particles (defined in the setUp of the base class)
    are constrained to move on a circle of radius 2./pi. 

    Four particles start the motion from the points ENWS and after one 
    second, the positions should be NWSE respectively.

    """
    def setup(self):
        self.integrator = solver.RK4Integrator(particles=self.particles,
                                               calcs = self.calcs)

    def test_constructor(self):
        """ Some constructor tests """
        self.assertEqual(self.integrator.nsteps, 4)
        self.assertEqual(self.integrator.calling_sequence, [])
        
    def test_motion(self):
        """ Perform the integration of the particle positons 

        The scheme is the RK2 integrator which is first order accurate 
        in time. The time step used for the integration is 1e-3 and thus
        we expect the positions of the particles to be exact to within 
        four decimal places.       

        """

        #setup the integrator

        self.integrator.setup_integrator()

        #set the time constants

        t = 0; tf = 1.0; dt = 1e-3
        
        integrator = self.integrator
        particles = integrator.particles
        pa = particles.arrays[0]

        original_pos = [(pa.x[i] ,pa.y[i]) for i in range(len(pa.x))]
        exact = (0.0, self.r), (-self.r, 0.0), (0.0, -self.r), (self.r, 0.0)

        while t <= tf:
            t += dt
            particles.update()
            integrator.integrate(dt)

        new_pos = [(pa.x[i] ,pa.y[i]) for i in range(len(pa.x))]
        self.print_pos('RK4 Integration', original_pos, new_pos)

        for i in range(4):
            self.assertAlmostEqual(new_pos[i][0], exact[i][0], 12)
            self.assertAlmostEqual(new_pos[i][1], exact[i][1], 12)

##############################################################################

class TestPredictorCorrectorIntegrator(IntegratorTestCase):
    """ Test for the Euler Integrator

    For the test, the particles (defined in the setUp of the base class)
    are constrained to move on a circle of radius 2./pi. 

    Four particles start the motion from the points ENWS and after one 
    second, the positions should be NWSE respectively.

    """
    def setup(self):
        self.integrator = solver.PredictorCorrectorIntegrator(
            particles=self.particles, calcs = self.calcs)

    def test_motion(self):
        """ Perform the integration of the particle positons 

        The scheme is the RK2 integrator which is first order accurate 
        in time. The time step used for the integration is 1e-3 and thus
        we expect the positions of the particles to be exact to within 
        four decimal places.       

        """

        #setup the integrator

        self.integrator.setup_integrator()

        #set the time constants

        t = 0; tf = 1.0; dt = 1e-3
        
        integrator = self.integrator
        particles = integrator.particles
        pa = particles.arrays[0]

        original_pos = [(pa.x[i] ,pa.y[i]) for i in range(len(pa.x))]
        exact = (0.0, self.r), (-self.r, 0.0), (0.0, -self.r), (self.r, 0.0)

        while t <= tf:
            t += dt
            particles.update()
            integrator.integrate(dt)

        new_pos = [(pa.x[i] ,pa.y[i]) for i in range(len(pa.x))]
        self.print_pos('Predictor Corrector', original_pos, new_pos)

        for i in range(4):
            self.assertAlmostEqual(new_pos[i][0], exact[i][0], 6)
            self.assertAlmostEqual(new_pos[i][1], exact[i][1], 6)

##############################################################################

class TestLeapFrogIntegrator(IntegratorTestCase):
    """ Test for the Euler Integrator

    For the test, the particles (defined in the setUp of the base class)
    are constrained to move on a circle of radius 2./pi. 

    Four particles start the motion from the points ENWS and after one 
    second, the positions should be NWSE respectively.

    """
    def setup(self):
        self.integrator = solver.LeapFrogIntegrator(
            particles=self.particles, calcs = self.calcs)

    def test_constructor(self):
        """ Some constructor tests """
        self.assertEqual(self.integrator.nsteps, 2)
        self.assertEqual(self.integrator.calling_sequence, [])

    def test_motion(self):
        """ Perform the integration of the particle positons 

        The scheme is the RK2 integrator which is first order accurate 
        in time. The time step used for the integration is 1e-3 and thus
        we expect the positions of the particles to be exact to within 
        four decimal places.       

        """

        #setup the integrator

        self.integrator.setup_integrator()

        #set the time constants

        t = 0; tf = 1.0; dt = 1e-3
        
        integrator = self.integrator
        particles = integrator.particles
        pa = particles.arrays[0]

        original_pos = [(pa.x[i] ,pa.y[i]) for i in range(len(pa.x))]
        exact = (0.0, self.r), (-self.r, 0.0), (0.0, -self.r), (self.r, 0.0)

        while t <= tf:
            t += dt
            particles.update()
            integrator.integrate(dt)

        new_pos = [(pa.x[i] ,pa.y[i]) for i in range(len(pa.x))]
        self.print_pos('Leap Frog', original_pos, new_pos)

        for i in range(4):
            self.assertAlmostEqual(new_pos[i][0], exact[i][0], 6)
            self.assertAlmostEqual(new_pos[i][1], exact[i][1], 6)

##############################################################################

if __name__ == '__main__':
    unittest.main()
