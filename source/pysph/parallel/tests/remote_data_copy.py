"""
Simple script to check if copies of remote data are properly done.
"""

# mpi import
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()

if num_procs > 3:
    raise SystemError, 'Start this script with 3 processors'
rank = comm.Get_rank()

# logging setup
import logging
logger = logging.getLogger()
log_file_name = '/tmp/log_pysph_'+str(rank)
logging.basicConfig(level=logging.DEBUG, filename=log_file_name,
                    filemode='w')
logger.addHandler(logging.StreamHandler())

# local imports
from pysph.base.particle_array import ParticleArray
from pysph.parallel.parallel_cell import ParallelCellManager
from pysph.solver.basic_generators import LineGenerator
from pysph.base.cell import INT_INF
from pysph.base.point import *

pcm = ParallelCellManager(initialize=False)

# create two particles, one with proc 0 another with proc 1
if rank == 0:
    parray = ParticleArray()
    parray.add_property({'name':'x', 'data':[0.4]})
    parray.add_property({'name':'h', 'data':[0.1]})
elif rank == 1:
    parray = ParticleArray()
    parray.add_property({'name':'x', 'data':[1.2]})
    parray.add_property({'name':'h', 'data':[0.1]})
elif rank == 2:
    parray = ParticleArray()
    parray.add_property({'name':'x', 'data':[2.0]})
    parray.add_property({'name':'h', 'data':[0.1]})

parray.add_property({'name':'y'})
parray.add_property({'name':'z'})
parray.add_property({'name':'t'})
parray.align_particles()

logger.debug('%s, %s'%(parray.x, parray.t))

pcm.add_array_to_bin(parray)
pcm.initialize()

# set the 't' property in proc 0 to -1000 and proc 1 to 1000.
if rank == 0:
    parray.t[0] = 1000.
if rank ==  1:
    parray.t[0] = 2000.
if rank == 2:
    parray.t[0] = 3000.

# get remote data.
pcm.update_remote_particle_properties([['t']])

logger.debug('t is %s'%(parray.get('t', only_real_particles=False)))
