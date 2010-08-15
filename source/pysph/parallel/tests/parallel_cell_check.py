"""
Some checks for the parallel cell manager.

Run this script only with two processors.

mpiexec -n 2 python parallel_cell_check.py

"""

# mpi imports
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
if num_procs > 4:
    raise SystemError, 'Start this script on less than 5 processors'
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

# create 2 particles, one with proc 0 another with proc 1

lg = LineGenerator(parallel_spacing=0.5)

if rank == 0:
    lg.start_point.x = -5.0
    lg.start_point.y = lg.start_point.z = 0.0
    
    lg.end_point.x = lg.end_point.y = lg.end_point.z = 0.0
    
    x, y, z = lg.get_coords()

    logger.info('Num particles : %d'%(len(x)))

    parray = ParticleArray(name='p1',
                           x={'data':x},
                           y={'data':y},
                           z={'data':z},
                           h={'data':None, 'default':0.5})

elif rank == 1:
    lg.start_point.x = 0.5
    lg.start_point.y = lg.start_point.z = 0.0

    lg.end_point.x = 5.0
    lg.end_point.y = lg.end_point.z = 0.0

    x, y, z = lg.get_coords()
    logger.info('Num particles : %d'%(len(x)))

    parray = ParticleArray(name='p1',
                           x={'data':x},
                           y={'data':y},
                           z={'data':z},
                           h={'data':None, 'default':0.5})
elif rank == 2:
    lg.start_point.x = 5.5
    lg.start_point.y = lg.start_point.z = 0.0

    lg.end_point.x = 10.0
    lg.end_point.y = lg.end_point.z = 0.0

    x, y, z = lg.get_coords()
    logger.info('Num particles : %d'%(len(x)))

    parray = ParticleArray(name='p1',
                           x={'data':x},
                           y={'data':y},
                           z={'data':z},
                           h={'data':None, 'default':0.5})
elif rank == 3:
    lg.start_point.x = 10.5
    lg.start_point.y = lg.start_point.z = 0.0

    lg.end_point.x = 15.0
    lg.end_point.y = lg.end_point.z = 0.0

    x, y, z = lg.get_coords()
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
pcm.add_array_to_bin(parray)
pcm.initialize()

pcm.set_jump_tolerance(INT_INF())

# on processor 1 move all particles from cell (7, 5, 5) to cell (8, 5, 5).
print rank, len(pcm.cells_dict)
print rank, ('\n%d '%rank).join([str(c) for c  in pcm.cells_dict.values()])

if rank == 1:
    c_7_5_5 = pcm.cells_dict.get(IntPoint(7, 5, 5))
    logger.debug('Cell (7, 5, 5) is %s'%(c_7_5_5))
    indices = []
    c_7_5_5.get_particle_ids(indices)
    indices = indices[0]
    logger.debug('NuM particles in (7, 5, 5) is %d'%(indices.length))
    parr = pcm.arrays_to_bin[0]
    x, y, z = parr.get('x', 'y', 'z')
    print len(x), x
    print indices.length, indices.get_npy_array()
    for i in range(indices.length):
        x[indices[i]] += c_7_5_5.cell_size

    parr.set_dirty(True)

pcm.update_status()
logger.debug('Calling cell manager update')
logger.debug('Is dirty %s'%(pcm.is_dirty))
pcm.update()

#logger.debug('hierarchy :%s'%(pcm.hierarchy_list))
logger.debug('roots cells : %s'%(pcm.cells_dict))
logger.debug('num particles : %d'%(parray.get_number_of_particles()))
logger.debug('real particles : %d'%(parray.num_real_particles))
