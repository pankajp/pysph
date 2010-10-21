""" An example solving the Ellptical drop test case """

import numpy
import pysph.base.api as base

import pysph.sph.api as sph
import pysph.solver.api as solver

def elliptical_drop(tf=0.00076, dt=1e-5):
    
    #Get the initial distribution

    pa = solver.fluid_solver.get_circular_patch(name='fluid')

    #set the kernel

    #kernel = base.CubicSplineKernel(dim=2)
    #kernel = base.QuinticSplineKernel(dim=2)
    kernel = base.WendlandQuinticSplineKernel(dim=2)

    #create the particle neighbor locator

    particles = base.Particles(arrays=[pa])

    #set the solver

    #s = solver.FluidSolver(kernel, solver.PredictorCorrectorIntegrator)
    s = solver.FluidSolver(kernel, solver.EulerIntegrator)

    s.set_final_time(tf)
    s.set_time_step(dt)
    s.setup_integrator(particles)
    
    s.solve()

    pa = s.particles.get_named_particle_array("fluid")
    solver.savez("drop_pc" + str(tf)+'.npz', x=pa.x, y=pa.y, p=pa.p)

    return s

def exact_solution(tf=0.00076, dt=1e-4):
    """ Exact solution for the equations """
    
    A0 = 100
    a0 = 1.0

    t = 0.0

    theta = numpy.linspace(0,2*numpy.pi, 101)

    Anew = A0
    anew = a0

    while t <= tf:
        t += dt

        Aold = Anew
        aold = anew
        
        Anew = Aold +  dt*(Aold*Aold*(aold**4 - 1))/(aold**4 + 1)
        anew = aold +  dt*(-aold * Aold)

    dadt = Anew**2 * (anew**4 - 1)/(anew**4 + 1)
    po = 0.5*-anew**2 * (dadt - Anew**2)
        
    return anew*numpy.cos(theta), 1/anew*numpy.sin(theta), po


#############################################################################

if __name__ == '__main__':
    pa = elliptical_drop()
