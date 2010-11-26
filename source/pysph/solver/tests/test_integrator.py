""" Tests for the integrator """
import numpy

#pysph imports 
import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph

from pysph.solver.integrator import Integrator

import unittest

Fluids = base.ParticleType.Fluid
Solids = base.ParticleType.Solid

class IntegratorBaseTestCase(unittest.TestCase):
    """ Tests for the setup integrator and other non-integrating functions """
    
    def setUp(self):
        """ A dummy solver is created along the lines of the solver test """

        self.kernel = kernel = base.CubicSplineKernel(dim = 2)

        self.solver = s = solver.Solver(kernel, solver.EulerIntegrator)

        self.particles = base.Particles(arrays=[base.get_particle_array()])

        # Add the default operations

        s.add_operation(solver.SPHAssignment(
                
                sph.TaitEquation(1,1), on_types = [Fluids, Solids],
                updates=['p','cs'], id = 'eos')

                             )
                
        s.add_operation(solver.SPHSummationODE(
                
                sph.SPHDensityRate(), on_types=[Fluids, Solids],
                from_types=[Fluids, Solids], updates=['rho'], id='density_rate')
                             
                             )


        s.add_operation(solver.SPHSummationODE(
                
                sph.SPHPressureGradient(), on_types=[Fluids],
                from_types=[Fluids, Solids], updates=['u','v'], id='pgrad')

                             )

        s.add_operation(solver.SPHSummationODE(

                sph.MonaghanArtificialVsicosity(), on_types=[Fluids],
                from_types=[Fluids, Solids], updates=['u','v'], id='avisc')

                             )

        s.to_step([Fluids])
        s.set_xsph(eps = 0.5)

        s.setup_integrator(self.particles)
        
        self.integrator = s.integrator

    def test_setup_integrator(self):
        """ Test the setup integrator function """

        integrator = self.integrator

        integrator._setup_integrator()

        calcs = []
        calcs.extend(integrator.calcs)
        calcs.extend(integrator.pcalcs)

        ncalcs = len(calcs)

        initial_props = integrator.initial_props
        step_props = integrator.step_props
        k_props = integrator.k_props

        # Test the initial_props dictionary

        k_num = 'k1'
        for i in range(ncalcs):
            calc = calcs[i]

            updates = calc.updates

            iprops = [k+'_0' for k in updates]
            sprops = []
            kprops = []

            self.assertEqual(initial_props[calc.id], iprops)

            for j in range(len(updates)):
                update_prop = updates[j]

                sprops.append(update_prop+'_'+str(i) + str(j))

                kprops.append(k_num + '_' + update_prop + str(i) + str(j))

            self.assertEqual(step_props[calc.id], sprops)

            if calc.integrates:
                self.assertEqual(k_props[calc.id][k_num], kprops)


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
                                      from_types=[Fluids], on_types=[Fluids],
                                      updates=['x','y'], id = 'circlex')

        circley = solver.SPHSimpleODE(sph.MoveCircleY(), 
                                      from_types=[Fluids], on_types=[Fluids],
                                      updates=['x','y'], id = 'circley')
        
        self.calcx = circlex.get_calcs(particles, self.kernel)
        self.calcy = circley.get_calcs(particles, self.kernel)

        self.calcs = []
        self.calcs.extend(self.calcx)
        self.calcs.extend(self.calcy)

        self.integrator = Integrator(particles=particles, calcs=self.calcs)

        self.setup()

    def setup(self):
        pass

    def test_constructor(self):
        """ Some constructor tests """
        self.assertEqual(self.integrator.nsteps, 1)
        self.assertEqual(self.integrator.calling_sequence, [])

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

        self.integrator._setup_integrator()

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
            integrator._integrate(dt)
            
        new_pos = [(pa.x[i] ,pa.y[i]) for i in range(len(pa.x))]
        
        #self.print_pos('Euler Integration', original_pos, new_pos)

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
        
    def test_motion(self):
        """ Perform the integration of the particle positons 

        The scheme is the RK2 integrator which is first order accurate 
        in time. The time step used for the integration is 1e-3 and thus
        we expect the positions of the particles to be exact to within 
        four decimal places.       

        """

        #setup the integrator

        self.integrator._setup_integrator()

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
            integrator._integrate(dt)

        new_pos = [(pa.x[i] ,pa.y[i]) for i in range(len(pa.x))]

        #self.print_pos('RK2 Integration', original_pos, new_pos)

        for i in range(4):
            self.assertAlmostEqual(new_pos[i][0], exact[i][0], 6)
            self.assertAlmostEqual(new_pos[i][1], exact[i][1], 6)            

        
# ##############################################################################

# class TestRK4Integrator(IntegratorTestCase):
#     """ Test for the Euler Integrator

#     For the test, the particles (defined in the setUp of the base class)
#     are constrained to move on a circle of radius 2./pi. 

#     Four particles start the motion from the points ENWS and after one 
#     second, the positions should be NWSE respectively.

#     """
#     def setup(self):
#         self.integrator = solver.RK4Integrator(particles=self.particles,
#                                                calcs = self.calcs)

#     def test_constructor(self):
#         """ Some constructor tests """
#         self.assertEqual(self.integrator.nsteps, 4)
#         self.assertEqual(self.integrator.calling_sequence, [])
        
#     def test_motion(self):
#         """ Perform the integration of the particle positons 

#         The scheme is the RK2 integrator which is first order accurate 
#         in time. The time step used for the integration is 1e-3 and thus
#         we expect the positions of the particles to be exact to within 
#         four decimal places.       

#         """

#         #setup the integrator

#         self.integrator.setup_integrator()

#         #set the time constants

#         t = 0; tf = 1.0; dt = 1e-3
        
#         integrator = self.integrator
#         particles = integrator.particles
#         pa = particles.arrays[0]

#         original_pos = [(pa.x[i] ,pa.y[i]) for i in range(len(pa.x))]
#         exact = (0.0, self.r), (-self.r, 0.0), (0.0, -self.r), (self.r, 0.0)

#         while t <= tf:
#             t += dt
#             particles.update()
#             integrator.integrate(dt)

#         new_pos = [(pa.x[i] ,pa.y[i]) for i in range(len(pa.x))]
#         self.print_pos('RK4 Integration', original_pos, new_pos)

#         for i in range(4):
#             self.assertAlmostEqual(new_pos[i][0], exact[i][0], 12)
#             self.assertAlmostEqual(new_pos[i][1], exact[i][1], 12)

# ##############################################################################

# class TestPredictorCorrectorIntegrator(IntegratorTestCase):
#     """ Test for the Euler Integrator

#     For the test, the particles (defined in the setUp of the base class)
#     are constrained to move on a circle of radius 2./pi. 

#     Four particles start the motion from the points ENWS and after one 
#     second, the positions should be NWSE respectively.

#     """
#     def setup(self):
#         self.integrator = solver.PredictorCorrectorIntegrator(
#             particles=self.particles, calcs = self.calcs)

#     def test_motion(self):
#         """ Perform the integration of the particle positons 

#         The scheme is the RK2 integrator which is first order accurate 
#         in time. The time step used for the integration is 1e-3 and thus
#         we expect the positions of the particles to be exact to within 
#         four decimal places.       

#         """

#         #setup the integrator

#         self.integrator.setup_integrator()

#         #set the time constants

#         t = 0; tf = 1.0; dt = 1e-3
        
#         integrator = self.integrator
#         particles = integrator.particles
#         pa = particles.arrays[0]

#         original_pos = [(pa.x[i] ,pa.y[i]) for i in range(len(pa.x))]
#         exact = (0.0, self.r), (-self.r, 0.0), (0.0, -self.r), (self.r, 0.0)

#         while t <= tf:
#             t += dt
#             particles.update()
#             integrator.integrate(dt)

#         new_pos = [(pa.x[i] ,pa.y[i]) for i in range(len(pa.x))]
#         self.print_pos('Predictor Corrector', original_pos, new_pos)

#         for i in range(4):
#             self.assertAlmostEqual(new_pos[i][0], exact[i][0], 6)
#             self.assertAlmostEqual(new_pos[i][1], exact[i][1], 6)

# ##############################################################################

# class TestLeapFrogIntegrator(IntegratorTestCase):
#     """ Test for the Euler Integrator

#     For the test, the particles (defined in the setUp of the base class)
#     are constrained to move on a circle of radius 2./pi. 

#     Four particles start the motion from the points ENWS and after one 
#     second, the positions should be NWSE respectively.

#     """
#     def setup(self):
#         self.integrator = solver.LeapFrogIntegrator(
#             particles=self.particles, calcs = self.calcs)

#     def test_constructor(self):
#         """ Some constructor tests """
#         self.assertEqual(self.integrator.nsteps, 2)
#         self.assertEqual(self.integrator.calling_sequence, [])

#     def test_motion(self):
#         """ Perform the integration of the particle positons 

#         The scheme is the RK2 integrator which is first order accurate 
#         in time. The time step used for the integration is 1e-3 and thus
#         we expect the positions of the particles to be exact to within 
#         four decimal places.       

#         """

#         #setup the integrator

#         self.integrator.setup_integrator()

#         #set the time constants

#         t = 0; tf = 1.0; dt = 1e-3
        
#         integrator = self.integrator
#         particles = integrator.particles
#         pa = particles.arrays[0]

#         original_pos = [(pa.x[i] ,pa.y[i]) for i in range(len(pa.x))]
#         exact = (0.0, self.r), (-self.r, 0.0), (0.0, -self.r), (self.r, 0.0)

#         while t <= tf:
#             t += dt
#             particles.update()
#             integrator.integrate(dt)

#         new_pos = [(pa.x[i] ,pa.y[i]) for i in range(len(pa.x))]
#         self.print_pos('Leap Frog', original_pos, new_pos)

#         for i in range(4):
#             self.assertAlmostEqual(new_pos[i][0], exact[i][0], 6)
#             self.assertAlmostEqual(new_pos[i][1], exact[i][1], 6)

##############################################################################

if __name__ == '__main__':
    unittest.main()
