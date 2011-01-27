""" An example solving the Ellptical drop test case """

import pysph.base.api as base
import pysph.solver.api as solver

app = solver.Application()
app.process_command_line()

particles = app.create_particles(
    solver.fluid_solver.get_circular_patch, name='fluid', type=0)
    
s = solver.FluidSolver(base.CubicSplineKernel(dim=2), 
                       solver.EulerIntegrator)

s.set_final_time(0.00076)
s.set_time_step(1e-5)

app.set_solver(s)

s.post_step_functions.append(solver.PrintNeighborInformation(count=10,
                                                             path=app.path,
                                                             rank=app.rank))

app.run()



