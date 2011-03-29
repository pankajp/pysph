""" Test the integrator for the NBody problem defined as follows.


100 particles are distributed randomly in the cube [-1,1] X [-1,1] X
[-1,1] each with random mass as well. Each particle is created as a
seperate ParticleArray. The interaction between particles is governed
by the NBodyForce function defined in sph/funcs/external_funcs.pyx
given by:

..math::

\frac{D\vec{v}}{Dt} = \sum_{j=1}^{N}\frac{m_j}{\norm(x_j-x_i}^3 +
\eps} \vec{x_{ji}}

Particle positions are integrated as

\frac{D\vec{x}{Dt}} = \vec{v}


The NBodyForce implements an all neighbor search such that all
particle indices from the source array are returned. This basically
subverts the neighbor locator as it is never called.


"""

import pysph.solver.api as solver
import pysph.base.api as base
import pysph.sph.api as sph

Fluids = base.ParticleType.Fluid

import unittest
import numpy

eps = 1e-3

dt = 1e-4
tf = dt*1000

np = 100

x0 = numpy.random.random(np)*2.0 - 1.0
y0 = numpy.random.random(np)*2.0 - 1.0
z0 = numpy.random.random(np)*2.0 - 1.0

m = numpy.ones_like(x0)
h0 = numpy.ones_like(x0)

def get_particle_array_positions(parrays):

    x = [pa.x[0] for pa in parrays]
    y = [pa.y[0] for pa in parrays]
    z = [pa.z[0] for pa in parrays]

    return x,y,z

def get_particle_array_veocities(parrays):
    u = [pa.u[0] for pa in parrays]
    v = [pa.v[0] for pa in parrays]
    w = [pa.w[0] for pa in parrays]

    return u,v,w    

class IntegratorTestCase(unittest.TestCase):

    def setUp(self):

        parrays = []

        for i in range(np):
            x = numpy.array( [x0[i]] )
            y = numpy.array( [y0[i]] )
            z = numpy.array( [z0[i]] )
            h = numpy.array( [h0[i]] )

            mi = numpy.array( [m[i]] )

            name = 'array' + str(i)
            pa = base.get_particle_array(name=name, x=x, y=y, z=z, h=h, m=mi )

            parrays.append(pa)

        self.parrays = parrays

        self.particles = base.Particles(arrays=parrays)

        kernel = base.CubicSplineKernel(dim=3)

        self.solver = s = solver.Solver(kernel, solver.EulerIntegrator)

        # NBodyForce operation

        s.add_operation(solver.SPHIntegration(

            sph.NBodyForce.withargs(eps=eps), on_types=[Fluids],
            from_types=[Fluids], updates=['u','v','w'], id='nbody_force' )
            
                        )

        # position stepping
    
        s.to_step([Fluids])

        # time step and final time
        
        s.set_final_time(tf)
        s.set_time_step(dt)

        # setup the integrator

        s.setup_integrator( self.particles )

    def test_construction(self):
        """ Test the construction of the calcs and the operations.

        There should be np particle arrays and calcs, one for each
        array. The list of sources for each calc should be all the
        particle arrays.

        """
        
        integrator = self.solver.integrator
        particles = integrator.particles

        
        # there should be np particle arrays
        
        self.assertEqual( len(particles.arrays), np )

        # number of calcs = narrays * noperations

        calcs = integrator.calcs        
        self.assertEqual( len(calcs), np*2 )

        ncalcs = integrator.ncalcs
        pcalcs = integrator.pcalcs

        self.assertEqual( len(ncalcs), np )
        self.assertEqual( len(pcalcs), np )

        for i in range(np):

            # test the nbody force calc
            
            ncalc = ncalcs[i]

            self.assertTrue( ncalc.integrates )

            srcs = ncalc.sources
            funcs = ncalc.funcs
            dst = ncalc.dest

            dst_name = 'array'+str(i) 
            self.assertEqual( dst.name, dst_name )

            self.assertEqual( len(srcs), np )
            self.assertEqual( len(funcs), np ) 

            for j in range(np):
                src = srcs[j]
                func = funcs[j]

                src_name = 'array'+str(j) 

                self.assertEqual( src.name, src_name )

                self.assertEqual( func.dest.name, dst_name )
                self.assertEqual( func.source.name, src_name )


            # test the position stepping calcs

            pcalc = pcalcs[i]

            self.assertTrue( pcalc.integrates )

            srcs = ncalc.sources
            funcs = ncalc.funcs
            dst = ncalc.dest

            dst_name = 'array'+str(i) 
            self.assertEqual( dst.name, dst_name )

            self.assertEqual( len(srcs), np )
            self.assertEqual( len(funcs), np ) 

            for j in range(np):
                src = srcs[j]
                func = funcs[j]

                src_name = 'array'+str(j) 

                self.assertEqual( src.name, src_name )

                self.assertEqual( func.dest.name, dst_name )
                self.assertEqual( func.source.name, src_name )

    def test_euler_integration(self):
        """ Test the EulerIntegration of the system """

        s = self.solver

        s.switch_integrator( solver.EulerIntegrator )
        
        x, y, z = x0.copy(), y0.copy(), z0.copy()

        xe = x.copy()
        ye = y.copy()
        ze = z.copy()

        xp, yp, zp = get_particle_array_positions(self.parrays)

        for i in range(np):
            self.assertAlmostEqual( xp[i], xe[i], 10 )
            self.assertAlmostEqual( yp[i], ye[i], 10 )
            self.assertAlmostEqual( zp[i], ze[i], 10 )            

        ue = numpy.zeros_like(x)
        ve = numpy.zeros_like(x)
        we = numpy.zeros_like(x)
        
        up, vp, wp = get_particle_array_veocities(self.parrays)

        for i in range(np):        
            self.assertAlmostEqual( up[i], ue[i], 10 )
            self.assertAlmostEqual( vp[i], ve[i], 10 )
            self.assertAlmostEqual( wp[i], we[i], 10 )

        t = 0.0

        while t <= tf:

            # Euler Integration

            for i in range(np):

                ax = numpy.zeros_like(x)
                ay = numpy.zeros_like(x)
                az = numpy.zeros_like(x)

                for j in range(np):

                    if not ( i == j ):

                        dx = x[j] - x[i]
                        dy = y[j] - y[i]
                        dz = z[j] - z[i]

                        invr = 1.0/(numpy.sqrt(dx*dx + dy*dy + dz*dz ) + eps)
                        invr3 = invr*invr*invr
                    
                        ax[i] += m[j]*invr3 * dx
                        ay[i] += m[j]*invr3 * dy
                        az[i] += m[j]*invr3 * dz

                    else:
                        pass
            
                # step the velocities

                ue[i] += ax[i] * dt
                ve[i] += ay[i] * dt
                we[i] += az[i] * dt

                # step the positions

                xe[i] += ue[i] * dt
                ye[i] += ve[i] * dt
                ze[i] += we[i] * dt

            # compare with PySPH integration

            s.integrator.integrate(dt)
                
            xp, yp, zp = get_particle_array_positions(self.parrays)
            up, vp, wp = get_particle_array_veocities(self.parrays)

            for i in range(np):
                self.assertAlmostEqual( xp[i], xe[i], 10 )
                self.assertAlmostEqual( yp[i], ye[i], 10 )
                self.assertAlmostEqual( zp[i], ze[i], 10 )

                self.assertAlmostEqual( up[i], ue[i], 10 )
                self.assertAlmostEqual( vp[i], ve[i], 10 )
                self.assertAlmostEqual( wp[i], we[i], 10 )

            # copy the euler variables for the next step

            x = xe.copy()
            y = ye.copy()
            z = ze.copy()

            t += dt

            print t

if __name__ == '__main__':
    unittest.main()
