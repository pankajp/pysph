""" Test for the OpenCL integrators """

import pysph.solver.api as solver
import pysph.base.api as base
import pysph.sph.api as sph

from pysph.solver.cl_integrator import CLIntegrator

Fluid = base.ParticleType.Fluid

import numpy
import unittest

import pyopencl as cl

class CLIntegratorTestCase(unittest.TestCase):
    """ Test the CLEulerIntegrator """

    def runTest(self):
        pass

    def setUp(self):
        """ The setup consists of two fluid particle arrays, each
        having one particle. The fluids are acted upon by an external
        vector force and gravity.

        Comparison is made with the PySPH integration of the system.
        
        """

        x1 = numpy.array([-0.5,])
        y1 = numpy.array([1.0, ])

        x2 = numpy.array([0.5,])
        y2 = numpy.array([1.0,])

        tmpx1 = numpy.ones_like(x1)
        tmpx2 = numpy.ones_like(x2)

        self.f1 = base.get_particle_array(name="fluid1", x=x1, y=y1,tmpx=tmpx1)
        self.f2 = base.get_particle_array(name="fluid2", x=x2, y=y2,tmpx=tmpx2)

        self.particles = base.Particles(arrays=[self.f1,self.f2])
        self.kernel = kernel = base.CubicSplineKernel(dim=2)

        gravity = solver.SPHIntegration(

            sph.GravityForce.withargs(gy=-10.0), on_types=[Fluid],
            updates=['u','v'], id='gravity'

            )

        force = solver.SPHIntegration(

            sph.GravityForce.withargs(gx = -10.0), on_types=[Fluid],
            updates=['u','v'], id='force'

            )

        position = solver.SPHIntegration(

            sph.PositionStepping, on_types=[Fluid],
            updates=['x','y'], id='step',

            )                    
        
        gravity.calc_type = sph.CLCalc
        force.calc_type = sph.CLCalc
        position.calc_type = sph.CLCalc

        gravity_calcs = gravity.get_calcs(self.particles, kernel)
        force_calcs = force.get_calcs(self.particles, kernel)
        position_calcs = position.get_calcs(self.particles, kernel)

        self.calcs = calcs = []
        calcs.extend(gravity_calcs)
        calcs.extend(force_calcs)
        calcs.extend(position_calcs)

        self.integrator = CLIntegrator(self.particles, calcs)

        self.ctx = ctx = cl.create_some_context()
        self.integrator.setup_integrator(ctx)
        self.queue = calcs[0].queue

        self.dt = 0.1
        self.nsteps = 10

    def test_setup_integrator(self):
        """ Test the construction of the integrator """

        integrator = self.integrator
        calcs = integrator.calcs

        self.assertEqual( len(calcs), 6 )
        for calc in calcs:
            self.assertTrue( isinstance(calc, sph.CLCalc) )

        # check that setup_cl has been called for the arrays

        self.assertTrue(self.f1.cl_setup_done)
        self.assertTrue(self.f2.cl_setup_done)

        # check for the additional properties created by the integrator

        for arr in [self.f1, self.f2]:

            # Initial props
            self.assertTrue( arr.properties.has_key('x_0') )
            self.assertTrue( arr.properties.has_key('y_0') )
            self.assertTrue( arr.properties.has_key('u_0') )
            self.assertTrue( arr.properties.has_key('v_0') )
            
            self.assertTrue( arr.cl_properties.has_key('cl_x_0') )
            self.assertTrue( arr.cl_properties.has_key('cl_y_0') )
            self.assertTrue( arr.cl_properties.has_key('cl_u_0') )
            self.assertTrue( arr.cl_properties.has_key('cl_v_0') )

        # check for the k1 step props

        arr = self.f1

        self.assertTrue( arr.properties.has_key('k1_u00') )
        self.assertTrue( arr.properties.has_key('k1_v01') )

        self.assertTrue( arr.properties.has_key('k1_u20') )
        self.assertTrue( arr.properties.has_key('k1_v21') )

        self.assertTrue( arr.properties.has_key('k1_x40') )
        self.assertTrue( arr.properties.has_key('k1_y41') )

        self.assertTrue( arr.cl_properties.has_key('cl_k1_u00') )
        self.assertTrue( arr.cl_properties.has_key('cl_k1_v01') )

        self.assertTrue( arr.cl_properties.has_key('cl_k1_u20') )
        self.assertTrue( arr.cl_properties.has_key('cl_k1_v21') )

        self.assertTrue( arr.cl_properties.has_key('cl_k1_x40') )
        self.assertTrue( arr.cl_properties.has_key('cl_k1_y41') )

class CLEulerIntegratorTestCase(CLIntegratorTestCase):
    """ Test the Euler Integration of the system using OpenCL """

    def reference_euler_solution(self, x, y, u, v):
        """ Get the reference solution:

        X = X + h*dt

        """
        dt = self.dt
        fx = -10.0
        fy = -10.0

        x += u*dt
        y += v*dt

        u += fx*dt
        v += fy*dt

        return x, y, u, v

    def test_integrate(self):

        # set the integrator type
        self.integrator = solver.CLEulerIntegrator(self.particles, self.calcs)
        self.integrator.setup_integrator(self.ctx)

        integrator = self.integrator
        f1 = self.f1
        f2 = self.f2

        nsteps = 100

        for i in range(nsteps):
            integrator.integrate(self.dt)

            f1.read_from_buffer()
            f2.read_from_buffer()

            f1x, f1y, f1u, f1v = self.reference_euler_solution(f1.x, f1.y,
                                                               f1.u, f1.v)

            f2x, f2y, f2u, f2v = self.reference_euler_solution(f2.x, f2.y,
                                                               f2.u, f2.v)

            self.assertAlmostEqual(f1.x, f1x, 8)
            self.assertAlmostEqual(f1.y, f1y, 8)
            self.assertAlmostEqual(f1.u, f1u, 8)
            self.assertAlmostEqual(f1.v, f1v, 8)

            self.assertAlmostEqual(f2.x, f2x, 8)
            self.assertAlmostEqual(f2.y, f2y, 8)
            self.assertAlmostEqual(f2.u, f2u, 8)
            self.assertAlmostEqual(f2.v, f2v, 8)
            
if __name__ == '__main__':
    unittest.main()
