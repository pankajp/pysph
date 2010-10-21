""" An example script for running the shock tube problem using Standard
SPH.

Global properties for the shock tube problem:
---------------------------------------------
x ~ [-.6,.6], dxl = 0.001875, dxr = dxl*4, m = dxl, h = 2*dxr
rhol = 1.0, rhor = 0.25, el = 2.5, er = 1.795, pl = 1.0, pr = 0.1795


These are obtained from the solver.shock_tube_solver.standard_shock_tube_data
"""
import logging

import pysph.solver.api as solver
from pysph.base.kernels import CubicSplineKernel

# Create the application.
app = solver.Application()
app.setup_logging(filename='shock_tube.log', loglevel=logging.INFO)
app.process_command_line()
particles = app.create_particles(solver.shock_tube_solver.standard_shock_tube_data,
                                 name='fluid', type=0)

# Choose the kernel 
kernel = CubicSplineKernel(dim=1)
# Set the solver up.
s = solver.ShockTubeSolver(kernel, solver.EulerIntegrator)
# set the solver constants.
s.set_final_time(0.045)#(0.15)
s.set_time_step(3e-4)
# Set the application's solver.
app.set_solver(s)

# Run the application.
app.run()

# Once that is done, save the output.  This could use a lot more work.
pa = particles.arrays[0]
solver.savez_compressed('shock_tube_'+str(app.rank)+'.npz', 
                         x=pa.x, p=pa.p, rho=pa.rho, u=pa.u, e=pa.e)

