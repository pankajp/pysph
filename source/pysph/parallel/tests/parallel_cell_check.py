""" Some checks for the parallel cell manager.

Run this script only with less than 5 processors.
example : mpiexec -n 2 python parallel_cell_check.py
"""

import time

# mpi imports
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
#if num_procs > 4:
#    raise SystemError, 'Start this script on less than 5 processors'
rank = comm.Get_rank()

# logging setup
import logging
logger = logging.getLogger()
log_file_name = 'parallel_cell_check.log.'+str(rank)
logging.basicConfig(level=logging.DEBUG, filename=log_file_name,
                    filemode='w')
logger.addHandler(logging.StreamHandler())

# local imports
from pysph.base.particle_array import ParticleArray
from pysph.parallel.parallel_cell import ParallelCellManager
from pysph.solver.basic_generators import LineGenerator
from pysph.base.cell import INT_INF
from pysph.base.point import *

from pysph.parallel.load_balancer import LoadBalancer

pcm = ParallelCellManager(initialize=False)

# create 2 particles, one with proc 0 another with proc 1

lg = LineGenerator(particle_spacing=0.5)

lg.start_point.x = 0.0
lg.end_point.x = 10.0
lg.start_point.y = lg.start_point.z = 0.0
lg.end_point.y = lg.end_point.z = 0.0

x, y, z = lg.get_coords()
num_particles = len(x)

logger.info('Num particles : %d'%(len(x)))

parray = ParticleArray(name='p1',
                       x={'data':x},
                       y={'data':y},
                       z={'data':z},
                       h={'data':None, 'default':0.5})


# add parray to the cell manager
parray.add_property({'name':'u'})
parray.add_property({'name':'v'})
parray.add_property({'name':'w'})
parray.add_property({'name':'rho'})
parray.add_property({'name':'p'})

parray = LoadBalancer.distribute_particles(parray, num_procs, 1.0)[rank]
pcm.add_array_to_bin(parray)

np = pcm.arrays_to_bin[0].num_real_particles
nptot = comm.bcast(comm.reduce(np))
assert nptot == num_particles

pcm.initialize()

np = pcm.arrays_to_bin[0].num_real_particles
nptot = comm.bcast(comm.reduce(np))
assert nptot == num_particles

pcm.set_jump_tolerance(INT_INF())

logger.debug('%d: num_cells=%d'%(rank,len(pcm.cells_dict)))
logger.debug('%d:'%rank + ('\n%d '%rank).join([str(c) for c  in pcm.cells_dict.values()]))

# on processor 0 move all particles from one of its cell to the next cell
if rank == 0:
    cell = pcm.cells_dict.get(list(pcm.proc_map.cell_map.values()[0])[0])
    logger.debug('Cell is %s'%(cell))
    indices = []
    cell.get_particle_ids(indices)
    indices = indices[0]
    logger.debug('Num particles in Cell is %d'%(indices.length))
    parr = cell.arrays_to_bin[0]
    x, y, z = parr.get('x', 'y', 'z', only_real_particles=False)
    logger.debug(str(len(x)) + str(x))
    logger.debug(str(indices.length) + str(indices.get_npy_array()))
    for i in range(indices.length):
        x[indices[i]] += cell.cell_size

    parr.set_dirty(True)

pcm.update_status()
logger.debug('Calling cell manager update')
logger.debug('Is dirty %s'%(pcm.is_dirty))
pcm.update()

np = pcm.arrays_to_bin[0].num_real_particles
nptot = comm.bcast(comm.reduce(np))
assert nptot == num_particles

#logger.debug('hierarchy :%s'%(pcm.hierarchy_list))
logger.debug('cells : %s'%(pcm.cells_dict))
logger.debug('num particles : %d'%(parray.get_number_of_particles()))
logger.debug('real particles : %d'%(parray.num_real_particles))
