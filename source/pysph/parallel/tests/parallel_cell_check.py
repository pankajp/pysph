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
pcm.add_array_to_bin(parray)
pcm.initialize()
