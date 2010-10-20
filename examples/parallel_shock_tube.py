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
import pysph.sph.api as sph
import pysph.solver.api as solver
from pysph.base.carray import LongArray
from pysph.base.particles import Particles, get_particle_array
from pysph.base.kernels import CubicSplineKernel

comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

# logging setup
logger = logging.getLogger()
log_file_name = '/tmp/log_pysph_'+str(rank)
logging.basicConfig(level=logging.DEBUG, filename=log_file_name,
                    filemode='w')
logger.addHandler(logging.StreamHandler())

dxl = 0.001875
dxr = dxl*4

#get the distributed particle arrays

pa = solver.shock_tube_solver.standard_shock_tube_data(name='fluid', type=0)
pa = solver.get_distributed_particles(pa, comm, cell_size=dxr)

#choose the kernel 
kernel = CubicSplineKernel(dim=1)

#create the particles in parallel without load balancing

particles = Particles(arrays=[pa], in_parallel=True, load_balancing=True)

#set the solver with the partices, kernel and EulerIntegrator

s = solver.ShockTubeSolver(kernel, solver.EulerIntegrator)

#set the solver constants and begin solve

s.set_final_time(0.15)
s.set_time_step(3e-4)
s.setup_integrator(particles)

#numpy.savez('shock_tube_initial_'+str(rank)+'.npz', x=pa.x, p=pa.p, rho=pa.rho)
  
s.solve()

numpy.savez('shock_tube_'+str(rank)+'.npz', x=pa.x, p=pa.p, rho=pa.rho,
            u=pa.u, e=pa.e)
            
num_particles = pa.num_real_particles
logger.info("Number of real particles on %d are %d"%(rank, num_particles))
