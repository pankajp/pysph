""" An example dam break problem """

import numpy
import pysph.base.api as base
import pysph.sph.api as sph
import pysph.solver.api as solver

base.Fluids = base.ParticleType.Fluid
base.Solids = base.ParticleType.Solid

def get_dam():
    """ Get the particles corresponding to the dam and fluids """
    
    #First get the dam particles
    x,y = numpy.mgrid[0:3.2196:0.05, 0:1+1e-4:0.05]

    x = x.ravel()
    y = y.ravel()

    h = numpy.ones_like(x)*0.1
    m = numpy.ones_like(x)*0.05**2 * 1000

    n = len(x)

    type = numpy.zeros_like(x)
    
    #Tag the solid particles
    for i in range(n):
        if x[i] == max(x) or x[i] == min(x):
            type[i] = 1
        elif y[i] == max(y) or y[i] == min(y):
            type[i] = 1

    pa = base.get_particle_array(x=x,y=y,h=h,m=m)
    pa.set(**{'type':type})

    indx = []

    #Remove particles that are too close to the wall
    for i in range(n):
        if not type[i] == 1:
            if (0.2 - x[i]) >1e-10 or (x[i] - 2.2) > 1e-10:
                indx.append(i)

            elif (0.2-y[i]) > 1e-10 or (y[i] - 0.7) >1e-10:
                indx.append(i)

    la = base.LongArray(len(indx))
    la.set_data(numpy.array(indx))

    pa.remove_particles(la)

    #Set the density and pressure
    tmp = 1000*9.81/50
    rho = numpy.ones_like(pa.x)
    for i in range(len(pa.x)):
        if pa.type[i] == 0:
            rho[i] = numpy.power(1 + tmp*(0.7-pa.y[i]), (1./7))
            
    rho *= 1000
    p = 50*((rho/1000)**7.0 - 1) 
 
    pa.set(p=p, rho=rho)

    return pa

def dam_break(tf=1, dt=1e-4):
    
    #get the particle distribution
    pa = get_dam()

    np = pa.get_number_of_particles()

    #create the particle neighbor locator
    particles = base.Particles(pa=pa, kfac=2)

    #set the kernel
    kernel = base.CubicSplineKernel(dim=2)


    #create the solver
    s = solver.Solver(particles, kernel, solver.EulerIntegrator)

    s.add_operation(

        solver.SPHAssignment(sph.TaitEquation(pa, ko=50.0, ro=1000.0),
                             from_types=[base.Fluids],  on_types=[base.Fluids],
                             updates=['p'], id='eos')
        
        )

    s.add_operation(

        solver.SPHSummationODE(sph.SPHDensityRate(pa),
                               from_types=[base.Fluids], on_types=[base.Fluids],
                               updates=['rho'], id='density')
        )
                                           

    s.add_operation(

        solver.SPHSummationODE(sph.MomentumEquation(pa, sound_speed=50.0),
                               from_types=[base.Fluids],on_types=[base.Fluids], 
                               updates=['u','v'], id='mom')
        
        )
    
        
    s.add_operation(
        
        solver.SPHSummationODE(sph.BeckerBoundaryForce(pa, sound_speed=50.0),
                               from_types=[base.Solids], on_types=[base.Fluids],
                               updates=['u','v'], id='boundary')

        )

    s.add_operation(
        
        solver.SPHSimpleODE(sph.GravityForce(pa, gy = -9.81),
                            from_types=[base.Fluids], on_types=[base.Fluids],
                            updates=['u','v'], id='gravity')

        )


    s.add_operation(

        solver.SPHSummationODE(sph.XSPHCorrection(pa, eps=0.5),
                               from_types=[base.Fluids], on_types=[base.Fluids],
                               updates=['x','y'], id='xsph')

        )


    s.add_operation(
        
        solver.SPHSimpleODE(sph.PositionStepping(pa),
                            from_types=[base.Fluids], on_types=[base.Fluids],
                            updates=['x','y'], id='step')

        )

    s.set_final_time(tf)
    s.set_time_step(dt)
    s.setup_integrator()
    
    s.solve()

    return s

if __name__ == '__main__':
    dam_break(tf=5)
