""" Initialization routines for shock tube problems """

#import pysph.base.api as base
import pysph.base.api as base
import pysph.sph.api as sph
import pysph.solver.api as solver

Fluids = base.ParticleType.Fluid

def shock_tube_with_xsph(tf = 0.15, dt=3e-4, eps=0.5):
    """ Solve the Standard Shock Tube problem """
    
    #define the kernel to use

    kernel = base.CubicSplineKernel(dim=1)

    #set the solver

    s = solver.ShockTubeSolver(kernel, solver.EulerIntegrator)

    #define the operation and insert it in the appropriate location

    s.add_operation(solver.SPHSummationODE(sph.XSPHCorrection(eps=eps),
                                           from_types=[Fluids],
                                           on_types=[Fluids],
                                           updates=['x'], id='xsph'
                                           ),
                     id='step')

    shock = solver.shock_tube_solver.standard_shock_tube_data(name="fluid")
    particles = base.Particles([shock], in_parallel=False)
    s.setup_integrator(particles)
    
    s.set_final_time(tf)
    s.set_time_step(dt)

    s.solve()

    pa = s.particles.get_named_particle_array("fluid")
    solver.savez("shock_tube.npz", x=pa.x, p=pa.p, rho=pa.rho, u=pa.u,
                 e=pa.e)

    return s

if __name__ == '__main__':
    shock_tube_with_xsph(eps=0.0)
