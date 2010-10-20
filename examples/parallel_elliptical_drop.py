""" An example script for running the shock tube problem on two processors 

Global properties for the shock tube problem:
---------------------------------------------
x ~ [-.6,.6], dxl = 0.001875, dxr = dxl*4, m = dxl, h = 2*dxr
rhol = 1.0, rhor = 0.25, el = 2.5, er = 1.795, pl = 1.0, pr = 0.1795


Setup:
------
The two particle arrays on either side of the discontinuity are created
on separate processors.

processor 0:
   x ~ [-.6,0], dx = dxl = 0.001875, h = 4*dxl, m = dxl, rho = 1.0, e = 2.5
   p = (1.4-1)*rho*e = 1.0

processor 1:
   x ~ [0, .6], dx = dxr = 0.0075, h = 2*dxr, m = dxl, rho = 0.25, e = 1.795
   p = (1.4-1)*rho*e = 0.1795


Run the program as:
mpirun -n 2 python parallel_shock_tube.py

"""

# mpi imports
from mpi4py import MPI

#numpy and logging
import numpy, logging

#local pysph imports
import pysph.solver.api as solver
import pysph.base.api as base

comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

# logging setup
logger = logging.getLogger()
log_file_name = '/tmp/log_pysph_'+str(rank)
logging.basicConfig(level=logging.DEBUG, filename=log_file_name,
                    filemode='w')
logger.addHandler(logging.StreamHandler())

#get the distributed particle arrays

pa = solver.fluid_solver.get_circular_patch(name='fluid', type=0)
pa = solver.get_distributed_particles(pa, comm, cell_size=0.025)

#set the kernel

kernel = base.CubicSplineKernel(dim=2)

#create the particles in parallel without load balancing

particles = base.Particles(arrays=[pa], in_parallel=True, load_balancing=False)

#set the solver with the partices, kernel and EulerIntegrator

s = solver.FluidSolver(kernel, solver.EulerIntegrator)

#set the solver constants and begin solve

s.set_final_time(0.00076)
s.set_time_step(1e-5)
s.setup_integrator(particles)

solver.savez('drop_initial_'+str(rank)+'.npz', x=pa.x, y=pa.y)

s.solve()

solver.savez('drop_'+str(rank)+'.npz', x=pa.x, y=pa.y)
            
num_particles = pa.num_real_particles
logger.info("Number of real particles on %d are %d"%(rank, num_particles))
