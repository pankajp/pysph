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

# Create the application, do this first so the application sets up the
# logging and also gets all command line arguments.
app = solver.Application()
# Process command line args first, this also sets up the logging.
app.process_command_line()

# Create the particles automatically, the application calls a supplied
# function which generates the particles.
particles = app.create_particles(
    solver.shock_tube_solver.standard_shock_tube_data,
    name='fluid', type=0)
pa = particles.arrays[0]

# Set the solver up.
s = solver.ShockTubeSolver(CubicSplineKernel(dim=1), solver.EulerIntegrator)
# set the default solver constants.
s.set_final_time(0.15)
s.set_time_step(3e-4)

# Set the application's solver.  We do this at the end since the user
# may have asked for a different timestep/final time on the command
# line.
app.set_solver(s)

# Run the application.
app.run()

# Once application has run, save the output.  This could use a lot more work.
pa = particles.arrays[0]

