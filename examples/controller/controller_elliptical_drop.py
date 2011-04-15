""" An example solving the Elliptical drop test case with various interfaces """

import pysph.base.api as base
import pysph.solver.api as solver


app = solver.Application()
app.process_command_line(['-q', '--interactive',
                    '--xml-rpc=0.0.0.0:8900', '--multiproc=pysph@0.0.0.0:8800'])

particles = app.create_particles(False,
    solver.fluid_solver.get_circular_patch, name='fluid', type=0)

s = solver.FluidSolver(dim=2, integrator_type=solver.EulerIntegrator)

app.set_solver(s)
s.set_time_step(1e-5)
s.set_final_time(1e-1)
s.pfreq = 1000


if __name__ == '__main__':
    app.run()
    
