""" Shock tube problem with the ADKE procedure of Sigalotti """

import pysph.solver.api as solver
import pysph.base.api as base
import pysph.sph.api as sph

from pysph.base.kernels import CubicSplineKernel

Fluid = base.ParticleType.Fluid

# Shock tube parameters

dxl = 0.6/320
dxr = 4*dxl

h0 = 2*dxr
eps = 0.4
k = 0.7

g1 = 0.5
g2 = 0.5

# Create the application

app = solver.Application()
app.process_command_line()

particles = app.create_particles(
    variable_h=True, callable=solver.shock_tube_solver.standard_shock_tube_data,
    name='fluid', type=Fluid)

pa = particles.get_named_particle_array('fluid')
pa.add_property({'name':'rhop'})
pa.add_property({'name':'div'})

# ensure that the array 'q' is available

pa.add_property( {'name':'q', 'type':'double'} )

s = solver.ShockTubeSolver(dim=1, integrator_type=solver.EulerIntegrator)


# add the smoothing length update function as the first operation

s.add_operation(solver.SPHOperation(

    sph.ADKEPilotRho.withargs(h0=h0),
    on_types=[Fluid], from_types=[Fluid], updates=['rhop'], id='adke_rho'),

                before=True, id="density")

s.add_operation(solver.SPHOperation(

    sph.ADKESmoothingUpdate.withargs(h0=h0, k=k, eps=eps),
    on_types=[Fluid], updates=['h'], id='adke'),
                
                before=True, id="density")
                
# add the update conduction coefficient after the density calculation

s.add_operation(solver.SPHOperation(

    sph.VelocityDivergence.withargs(),
    on_types=[Fluid], from_types=[Fluid], updates=['div'], id='vdivergence'),

    before=False, id='density')

s.add_operation(solver.SPHOperation(

    sph.ADKEConductionCoeffUpdate.withargs(g1=g1, g2=g2),
    on_types=[Fluid], from_types=[Fluid], updates=['q'], id='qcoeff'),

    before=False, id='vdivergence')


# add the artificial heat term after the energy equation

s.add_operation(solver.SPHIntegration(

    sph.ArtificialHeat.withargs(eta=0.1), on_types=[Fluid], from_types=[Fluid],
    updates=['e'], id='aheat'),

    before=False, id="enr")

s.set_final_time(0.15)
s.set_time_step(3e-4)

app.set_solver(s)

app.run()
