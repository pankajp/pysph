""" A script to demonstrate the simplest of calculations in parallel 

Setup:
------
Two particle arrays are created on two separate processors with the 
following procerties:

processor 0:
   x ~ [0,1], dx = 0.1, h = 0.2, m = 0.1, fval = x*x

processor 1:
   x ~ [1.1, 2], dx = 0.1, h = 0.2, m = 0.1, fval = x*x

"""

# mpi imports
from mpi4py import MPI

#numpy and logging
import numpy, logging

#local pysph imports
import pysph.sph.api as sph
import pysph.solver.api as solver
from pysph.base.carray import LongArray
from pysph.base.api import Particles, get_particle_array
from pysph.base.kernels import CubicSplineKernel

comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

if num_procs > 2:
    raise SystemError, 'Start this script on less than 5 processors'

# logging setup
logger = logging.getLogger()
log_file_name = '/tmp/log_pysph_'+str(rank)
logging.basicConfig(level=logging.DEBUG, filename=log_file_name,
                    filemode='w')
logger.addHandler(logging.StreamHandler())

#create the particles on processor 0
if rank == 0:
    x = numpy.linspace(0,1,11)
    h = numpy.ones_like(x)*0.2
    m = numpy.ones_like(x)*0.1
    rho = numpy.ones_like(x)
    fval = x*x

#create the particles on processor 1
if rank == 1:
    x = numpy.linspace(1.1,2,10)
    h = numpy.ones_like(x)*0.2
    m = numpy.ones_like(x)*0.1
    rho = numpy.ones_like(x)
    fval = x*x

#create the particles in parallel without load balancing
kernel = CubicSplineKernel(dim=1)
pa = get_particle_array(x=x, h=h, m=m, fval=fval, rho=rho)
particles = Particles([pa], in_parallel=True,
                      load_balancing=False)

#make sure the particles need updating
particles.update()

#choose the function and the sph calc
func = sph.SPHRho(pa, pa)
calc = sph.SPHCalc(particles=particles, kernel=kernel, func=func,
                   updates=['rho'], integrates=False)

tmpx = pa.get('tmpx', only_real_particles=False)
logger.debug('tempx for all particles %s'%(tmpx))

#perform the summation density operation
calc.sph()

local = pa.get('local', only_real_particles=False)
logger.debug('Local indices for process %d are %s'%(rank, local))

#check for the density values on each processor
rho = pa.get('tmpx', only_real_particles=True)
logger.debug('Density for local particles on processor %d is %s '%(rank, rho))
