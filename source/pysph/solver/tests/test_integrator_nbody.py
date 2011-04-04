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

zeros = numpy.zeros_like

eps = 1e-3

dt = 1e-3
tf = dt*100

np = 50

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
        """ Test EulerIntegration of the system """

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


            ax_k1 = numpy.zeros_like(x)
            ay_k1 = numpy.zeros_like(x)
            az_k1 = numpy.zeros_like(x)

            u_k1 = numpy.zeros_like(x)
            v_k1 = numpy.zeros_like(x)
            w_k1 = numpy.zeros_like(x)

            # Euler Integration

            for i in range(np):
                for j in range(np):
                    if not ( i == j ):

                        dx = x[j] - x[i]
                        dy = y[j] - y[i]
                        dz = z[j] - z[i]

                        invr = 1.0/(numpy.sqrt(dx*dx + dy*dy + dz*dz ) + eps)
                        invr3 = invr*invr*invr
                    
                        ax_k1[i] += m[j]*invr3 * dx
                        ay_k1[i] += m[j]*invr3 * dy
                        az_k1[i] += m[j]*invr3 * dz

                    else:
                        pass

                u_k1[i] = ue[i]
                v_k1[i] = ve[i]
                w_k1[i] = we[i]
            
                # step the velocities

                ue[i] += ax_k1[i] * dt
                ve[i] += ay_k1[i] * dt
                we[i] += az_k1[i] * dt

                # step the positions

                xe[i] += u_k1[i] * dt
                ye[i] += v_k1[i] * dt
                ze[i] += w_k1[i] * dt

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

    def test_rk2_integration(self):
        """ Test RK2Integration of the system """

        s = self.solver

        s.switch_integrator( solver.RK2Integrator )
        
        x, y, z = x0.copy(), y0.copy(), z0.copy()

        xrk2 = x.copy()
        yrk2 = y.copy()
        zrk2 = z.copy()

        _zeros = numpy.zeros_like(x)

        urk2 = _zeros.copy(); vrk2 = _zeros.copy(); wrk2 = _zeros.copy()
        
        t = 0.0

        while t <= tf:

            x_initial = xrk2.copy()
            y_initial = yrk2.copy()
            z_initial = zrk2.copy()

            u_initial = urk2.copy()
            v_initial = vrk2.copy()
            w_initial = wrk2.copy()

            ax_k1 = numpy.zeros_like(x)
            ay_k1 = numpy.zeros_like(x)
            az_k1 = numpy.zeros_like(x)

            u_k1 = numpy.zeros_like(x)
            v_k1 = numpy.zeros_like(x)
            w_k1 = numpy.zeros_like(x)

            x_k1 = numpy.zeros_like(x)
            y_k1 = numpy.zeros_like(x)
            z_k1 = numpy.zeros_like(x)

            ax_k2 = numpy.zeros_like(x)
            ay_k2 = numpy.zeros_like(x)
            az_k2 = numpy.zeros_like(x)

            u_k2 = numpy.zeros_like(x)
            v_k2 = numpy.zeros_like(x)
            w_k2 = numpy.zeros_like(x)

            # RK2 K1 evaluation

            for i in range(np):
                for j in range(np):
                    if not ( i == j ):

                        dx = xrk2[j] - xrk2[i]
                        dy = yrk2[j] - yrk2[i]
                        dz = zrk2[j] - zrk2[i]

                        invr = 1.0/(numpy.sqrt(dx*dx + dy*dy + dz*dz ) + eps)
                        invr3 = invr*invr*invr

                        ax_k1[i] += m[j]*invr3 * dx
                        ay_k1[i] += m[j]*invr3 * dy
                        az_k1[i] += m[j]*invr3 * dz

                u_k1[i] = urk2[i]
                v_k1[i] = vrk2[i]
                w_k1[i] = wrk2[i]

                # step the variables

                urk2[i] = u_initial[i] + dt * ax_k1[i]
                vrk2[i] = v_initial[i] + dt * ay_k1[i]
                wrk2[i] = w_initial[i] + dt * az_k1[i]

                x_k1[i] = x_initial[i] + dt * u_k1[i]
                y_k1[i] = y_initial[i] + dt * v_k1[i]
                z_k1[i] = z_initial[i] + dt * w_k1[i]

            # K2 evalluation
            for i in range(np):
                for j in range(np):
                    if not ( i == j ):

                        dx = x_k1[j] - x_k1[i]
                        dy = y_k1[j] - y_k1[i]
                        dz = z_k1[j] - z_k1[i]

                        invr = 1.0/(numpy.sqrt(dx*dx + dy*dy + dz*dz ) + eps)
                        invr3 = invr*invr*invr

                        ax_k2[i] += m[j]*invr3 * dx
                        ay_k2[i] += m[j]*invr3 * dy
                        az_k2[i] += m[j]*invr3 * dz

                u_k2[i] = urk2[i]
                v_k2[i] = vrk2[i]
                w_k2[i] = wrk2[i]

                # final step for the variables

                urk2[i] = u_initial[i] + 0.5 * dt * (ax_k1[i] + ax_k2[i])
                vrk2[i] = v_initial[i] + 0.5 * dt * (ay_k1[i] + ay_k2[i])
                wrk2[i] = w_initial[i] + 0.5 * dt * (az_k1[i] + az_k2[i])

                xrk2[i] = x_initial[i] + 0.5 * dt * (u_k1[i] + u_k2[i])
                yrk2[i] = y_initial[i] + 0.5 * dt * (v_k1[i] + v_k2[i])
                zrk2[i] = z_initial[i] + 0.5 * dt * (w_k1[i] + w_k2[i])
                
            # compare with PySPH integration

            s.integrator.integrate(dt)
                
            xp, yp, zp = get_particle_array_positions(self.parrays)
            up, vp, wp = get_particle_array_veocities(self.parrays)

            for i in range(np):
                self.assertAlmostEqual( xp[i], xrk2[i], 10 )
                self.assertAlmostEqual( yp[i], yrk2[i], 10 )
                self.assertAlmostEqual( zp[i], zrk2[i], 10 )

                self.assertAlmostEqual( up[i], urk2[i], 10 )
                self.assertAlmostEqual( vp[i], vrk2[i], 10 )
                self.assertAlmostEqual( wp[i], wrk2[i], 10 )

            t += dt

            x = xrk2.copy()
            y = yrk2.copy()
            z = zrk2.copy()

    def test_rk4_integration(self):
        """ Test RK4Integrator of the system """

        s = self.solver

        s.switch_integrator( solver.RK4Integrator )
        
        x, y, z = x0.copy(), y0.copy(), z0.copy()

        xrk4 = x.copy()
        yrk4 = y.copy()
        zrk4 = z.copy()

        _zeros = numpy.zeros_like(x)

        urk4 = _zeros.copy(); vrk4 = _zeros.copy(); wrk4 = _zeros.copy()

        t = 0.0

        while t <= tf:

            x_initial = x.copy(); y_initial = y.copy()
            z_initial = z.copy()

            u_initial = urk4.copy(); v_initial = vrk4.copy()
            w_initial = wrk4.copy()

            ax_k1 = _zeros.copy(); ay_k1 = _zeros.copy(); az_k1 = _zeros.copy()
            ax_k2 = _zeros.copy(); ay_k2 = _zeros.copy(); az_k2 = _zeros.copy()
            ax_k3 = _zeros.copy(); ay_k3 = _zeros.copy(); az_k3 = _zeros.copy()
            ax_k4 = _zeros.copy(); ay_k4 = _zeros.copy(); az_k4 = _zeros.copy()

            u_k1 = _zeros.copy(); v_k1 = _zeros.copy(); w_k1 = _zeros.copy()
            u_k2 = _zeros.copy(); v_k2 = _zeros.copy(); w_k2 = _zeros.copy()
            u_k3 = _zeros.copy(); v_k3 = _zeros.copy(); w_k3 = _zeros.copy()
            u_k4 = _zeros.copy(); v_k4 = _zeros.copy(); w_k4 = _zeros.copy()

            x_k1 = _zeros.copy(); y_k1 = _zeros.copy(); z_k1 = _zeros.copy()
            x_k2 = _zeros.copy(); y_k2 = _zeros.copy(); z_k2 = _zeros.copy()
            x_k3 = _zeros.copy(); y_k3 = _zeros.copy(); z_k3 = _zeros.copy()

            # RK4 K1 evaluation

            for i in range(np):
                for j in range(np):
                    if not ( i == j ):

                        dx = x[j] - x[i]
                        dy = y[j] - y[i]
                        dz = z[j] - z[i]

                        invr = 1.0/(numpy.sqrt(dx*dx + dy*dy + dz*dz ) + eps)
                        invr3 = invr*invr*invr

                        ax_k1[i] += m[j]*invr3 * dx
                        ay_k1[i] += m[j]*invr3 * dy
                        az_k1[i] += m[j]*invr3 * dz

                u_k1[i] = urk4[i]
                v_k1[i] = vrk4[i]
                w_k1[i] = wrk4[i]

                # step the variables

                urk4[i] = u_initial[i] + 0.5 * dt * ax_k1[i]
                vrk4[i] = v_initial[i] + 0.5 * dt * ay_k1[i]
                wrk4[i] = w_initial[i] + 0.5 * dt * az_k1[i]

                x_k1[i] = x_initial[i] + 0.5 * dt * u_k1[i]
                y_k1[i] = y_initial[i] + 0.5 * dt * v_k1[i]
                z_k1[i] = z_initial[i] + 0.5 * dt * w_k1[i]

            # K2 evalluation

            for i in range(np):
                for j in range(np):
                    if not ( i == j ):

                        dx = x_k1[j] - x_k1[i]
                        dy = y_k1[j] - y_k1[i]
                        dz = z_k1[j] - z_k1[i]

                        invr = 1.0/(numpy.sqrt(dx*dx + dy*dy + dz*dz ) + eps)
                        invr3 = invr*invr*invr

                        ax_k2[i] += m[j]*invr3 * dx
                        ay_k2[i] += m[j]*invr3 * dy
                        az_k2[i] += m[j]*invr3 * dz

                u_k2[i] = urk4[i]
                v_k2[i] = vrk4[i]
                w_k2[i] = wrk4[i]

                # step the variables

                urk4[i] = u_initial[i] + 0.5 * dt * ax_k2[i]
                vrk4[i] = v_initial[i] + 0.5 * dt * ay_k2[i]
                wrk4[i] = w_initial[i] + 0.5 * dt * az_k2[i]

                x_k2[i] = x_initial[i] + 0.5 * dt * u_k2[i]
                y_k2[i] = y_initial[i] + 0.5 * dt * v_k2[i]
                z_k2[i] = z_initial[i] + 0.5 * dt * w_k2[i]

            # K3 evalluation

            for i in range(np):
                for j in range(np):
                    if not ( i == j ):

                        dx = x_k2[j] - x_k2[i]
                        dy = y_k2[j] - y_k2[i]
                        dz = z_k2[j] - z_k2[i]

                        invr = 1.0/(numpy.sqrt(dx*dx + dy*dy + dz*dz ) + eps)
                        invr3 = invr*invr*invr

                        ax_k3[i] += m[j]*invr3 * dx
                        ay_k3[i] += m[j]*invr3 * dy
                        az_k3[i] += m[j]*invr3 * dz

                u_k3[i] = urk4[i]
                v_k3[i] = vrk4[i]
                w_k3[i] = wrk4[i]

                # step the variables

                urk4[i] = u_initial[i] + dt * ax_k3[i]
                vrk4[i] = v_initial[i] + dt * ay_k3[i]
                wrk4[i] = w_initial[i] + dt * az_k3[i]

                x_k3[i] = x_initial[i] + dt * u_k3[i]
                y_k3[i] = y_initial[i] + dt * v_k3[i]
                z_k3[i] = z_initial[i] + dt * w_k3[i]

            # K4

            for i in range(np):
                for j in range(np):
                    if not ( i == j ):

                        dx = x_k3[j] - x_k3[i]
                        dy = y_k3[j] - y_k3[i]
                        dz = z_k3[j] - z_k3[i]
                        
                        invr = 1.0/(numpy.sqrt(dx*dx + dy*dy + dz*dz ) + eps)
                        invr3 = invr*invr*invr

                        ax_k4[i] += m[j]*invr3 * dx
                        ay_k4[i] += m[j]*invr3 * dy
                        az_k4[i] += m[j]*invr3 * dz

                u_k4[i] = urk4[i]
                v_k4[i] = vrk4[i]
                w_k4[i] = wrk4[i]

                # final step

                urk4[i] = u_initial[i] + (1.0/6.0) * dt * \
                          (ax_k1[i] + 2*ax_k2[i] + 2*ax_k3[i] + ax_k4[i])
                
                vrk4[i] = v_initial[i] + (1.0/6.0) * dt * \
                          (ay_k1[i] + 2*ay_k2[i] + 2*ay_k3[i] + ay_k4[i])
                
                wrk4[i] = w_initial[i] + (1.0/6.0) * dt * \
                          (az_k1[i] + 2*az_k2[i] + 2*az_k3[i] + az_k4[i])


                xrk4[i] = x_initial[i] + (1.0/6.0) * dt * \
                          (u_k1[i] + 2*u_k2[i] + 2*u_k3[i] + u_k4[i])

                yrk4[i] = y_initial[i] + (1.0/6.0) * dt * \
                          (v_k1[i] + 2*v_k2[i] + 2*v_k3[i] + v_k4[i])

                zrk4[i] = z_initial[i] + (1.0/6.0) * dt * \
                          (w_k1[i] + 2*w_k2[i] + 2*w_k3[i] + w_k4[i])

            # compare with PySPH integration

            s.integrator.integrate(dt)
                
            xp, yp, zp = get_particle_array_positions(self.parrays)
            up, vp, wp = get_particle_array_veocities(self.parrays)

            for i in range(np):
                self.assertAlmostEqual( xp[i], xrk4[i], 10 )
                self.assertAlmostEqual( yp[i], yrk4[i], 10 )
                self.assertAlmostEqual( zp[i], zrk4[i], 10 )

                self.assertAlmostEqual( up[i], urk4[i], 10 )
                self.assertAlmostEqual( vp[i], vrk4[i], 10 )
                self.assertAlmostEqual( wp[i], wrk4[i], 10 )

            # copy the euler variables for the next step

            x = xrk4.copy()
            y = yrk4.copy()
            z = zrk4.copy()

            t += dt

    def test_predictor_corrector_integrator(self):
        """ Test PredictorCorrector integration of the system """

        s = self.solver

        s.switch_integrator( solver.PredictorCorrectorIntegrator )
        
        x, y, z = x0.copy(), y0.copy(), z0.copy()

        xpc = x.copy()
        ypc = y.copy()
        zpc = z.copy()

        _zeros = numpy.zeros_like(x)

        upc = _zeros.copy(); vpc = _zeros.copy(); wpc = _zeros.copy()

        t = 0.0

        while t<= tf:

            x_initial = x.copy(); y_initial = y.copy()
            z_initial = z.copy()

            u_initial = upc.copy(); v_initial = vpc.copy()
            w_initial = wpc.copy()

            ax_k1 = _zeros.copy(); ay_k1 = _zeros.copy(); az_k1 = _zeros.copy()
            ax_k2 = _zeros.copy(); ay_k2 = _zeros.copy(); az_k2 = _zeros.copy()

            u_k1 = _zeros.copy(); v_k1 = _zeros.copy(); w_k1 = _zeros.copy()
            u_k2 = _zeros.copy(); v_k2 = _zeros.copy(); w_k2 = _zeros.copy()
            
            x_k1 = _zeros.copy(); y_k1 = _zeros.copy(); z_k1 = _zeros.copy()
            x_k2 = _zeros.copy(); y_k2 = _zeros.copy(); z_k2 = _zeros.copy()

            # K1
            for i in range(np):
                for j in range(np):
                    if not ( i == j ):

                        dx = x[j] - x[i]
                        dy = y[j] - y[i]
                        dz = z[j] - z[i]

                        invr = 1.0/(numpy.sqrt(dx*dx + dy*dy + dz*dz ) + eps)
                        invr3 = invr*invr*invr

                        ax_k1[i] += m[j]*invr3 * dx
                        ay_k1[i] += m[j]*invr3 * dy
                        az_k1[i] += m[j]*invr3 * dz

                u_k1[i] = upc[i]
                v_k1[i] = vpc[i]
                w_k1[i] = wpc[i]

                # step the variables

                upc[i] = u_initial[i] + 0.5 * dt * ax_k1[i]
                vpc[i] = v_initial[i] + 0.5 * dt * ay_k1[i]
                wpc[i] = w_initial[i] + 0.5 * dt * az_k1[i]

                x_k1[i] = x_initial[i] + 0.5 * dt * u_k1[i]
                y_k1[i] = y_initial[i] + 0.5 * dt * v_k1[i]
                z_k1[i] = z_initial[i] + 0.5 * dt * w_k1[i]

            # K2
            for i in range(np):
                for j in range(np):
                    if not ( i == j ):

                        dx = x_k1[j] - x_k1[i]
                        dy = y_k1[j] - y_k1[i]
                        dz = z_k1[j] - z_k1[i]

                        invr = 1.0/(numpy.sqrt(dx*dx + dy*dy + dz*dz ) + eps)
                        invr3 = invr*invr*invr

                        ax_k2[i] += m[j]*invr3 * dx
                        ay_k2[i] += m[j]*invr3 * dy
                        az_k2[i] += m[j]*invr3 * dz

                u_k2[i] = upc[i]
                v_k2[i] = vpc[i]
                w_k2[i] = wpc[i]

                # step the variables

                upc[i] = u_initial[i] + 0.5 * dt * ax_k2[i]
                vpc[i] = v_initial[i] + 0.5 * dt * ay_k2[i]
                wpc[i] = w_initial[i] + 0.5 * dt * az_k2[i]

                x_k2[i] = x_initial[i] + 0.5 * dt * u_k2[i]
                y_k2[i] = y_initial[i] + 0.5 * dt * v_k2[i]
                z_k2[i] = z_initial[i] + 0.5 * dt * w_k2[i]

                # Final step

                upc[i] = 2*upc[i] - u_initial[i]
                vpc[i] = 2*vpc[i] - v_initial[i]
                wpc[i] = 2*wpc[i] - w_initial[i]

                xpc[i] = 2*x_k2[i] - x_initial[i]
                ypc[i] = 2*y_k2[i] - y_initial[i]
                zpc[i] = 2*z_k2[i] - z_initial[i]

            # compare with PySPH integration

            s.integrator.integrate(dt)
                
            xp, yp, zp = get_particle_array_positions(self.parrays)
            up, vp, wp = get_particle_array_veocities(self.parrays)

            for i in range(np):
                self.assertAlmostEqual( xp[i], xpc[i], 10 )
                self.assertAlmostEqual( yp[i], ypc[i], 10 )
                self.assertAlmostEqual( zp[i], zpc[i], 10 )

                self.assertAlmostEqual( up[i], upc[i], 10 )
                self.assertAlmostEqual( vp[i], vpc[i], 10 )
                self.assertAlmostEqual( wp[i], wpc[i], 10 )

            # copy the euler variables for the next step

            x = xpc.copy()
            y = ypc.copy()
            z = zpc.copy()

            t += dt

if __name__ == '__main__':
    unittest.main()
