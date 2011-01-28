""" Tests for the boundary functions """

import unittest
import numpy

import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph

Fluid = base.ParticleType.Fluid
Solid = base.ParticleType.Solid

class BoundaryForceTestCase(unittest.TestCase):

    def runTest(self):
        pass
    
    def setUp(self):
        """ A simple simulation in 2D with boundary particles spaced with a
        distance dp. The fluid particles are in a row just above as shown
        in the fgure. 

                   dx
                  o   o   o   o   o
            x x x x x x x x x x x x x x x  

        Y
        |  Z
        | /
        |/___ X
            

        Expected Behavior:
        -------------------
        The Monaghan Boundary force is defned such that a particle moving 
        parallel to the wall experiences a constant force.
        Each fluid particle can be thought of successive instances of a 
        moving particle and hence each of them should experience the same
        force.

        """

        #fluid particle spacing
        self.dx = dx = 0.1
        
        #solid particle spacing
        self.dp = dp = 0.05

        #the fluid properties
        xf = numpy.array([-.2,-.1,0.0, 0.1, 0.2])
        yf = numpy.array([dp, dp, dp, dp, dp])
        hf = numpy.ones_like(xf) * 2 * dx
        mf = numpy.ones_like(xf) * dx
        cs = numpy.ones_like(xf)
        rhof = numpy.ones_like(xf)

        self.fluid = base.get_particle_array(x=xf, y=yf, h=hf, m=mf, rho=rhof,
                                             cs=cs,
                                             name='fluid', type=Fluid)


        l  = base.Line(base.Point(-0.35), 0.70, 0.0)
        g = base.Geometry('line', lines=[l], is_closed=False)
        g.mesh_geometry(dp)
        self.solid = g.get_particle_array(re_orient=True)
        

        self.particles = particles = base.Particles(arrays=[self.fluid,
                                                            self.solid])

        self.kernel = kernel = base.CubicSplineKernel(dim=2)
        
        self.solver = solver.Solver(kernel, solver.EulerIntegrator)

        self.solver.add_operation(solver.SPHSummationODE(
                
                sph.MonaghanBoundaryForce(delp=dp),
                from_types = [Solid], on_types=[Fluid],
                updates=['u','v'], id='boundary')
                                  
                                  )
                             
        self.solver.setup_integrator(particles)

        self.integrator = self.solver.integrator

    def test_constructor(self):
        
        fluid = self.fluid

        self.assertTrue(numpy.allclose(fluid.u, 0.0))
        self.assertTrue(numpy.allclose(fluid.v, 0.0))
        self.assertTrue(numpy.allclose(fluid.cs, 1.0))

        
        solid = self.solid

        self.assertTrue(numpy.allclose(solid.nx, 0.0))
        self.assertTrue(numpy.allclose(solid.ny, 1.0))
        self.assertTrue(numpy.allclose(solid.nz, 0.0))

        self.assertTrue(numpy.allclose(solid.tx, 1.0))
        self.assertTrue(numpy.allclose(solid.ty, 0.0))
        self.assertTrue(numpy.allclose(solid.tz, 0.0))

    def test_force(self):
        
        fluid = self.particles.get_named_particle_array('fluid')
        np = fluid.get_number_of_particles()
        calc = self.integrator.calcs[0]

        # evaluate the force and store the result in tmp

        self.particles.update()

        calc.sph('tmpx', 'tmpy', 'tmpz')

        force = fluid.tmpy

        for i in range(np):
            self.assertTrue(numpy.allclose(force, force[i]))
                        
if __name__ == '__main__':
    unittest.main()
