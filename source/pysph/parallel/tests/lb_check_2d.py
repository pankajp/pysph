"""
Simple scripy to check if the load balancing works on 2-d data.
"""

# mpi imports
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

# logging setup
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
from pysph.solver.basic_generators import RectangleGenerator
from pysph.base.cell import INT_INF
from pysph.base.point import *


pcm = ParallelCellManager(initialize=False, dimension=2)

parray = ParticleArray(name='parray')

if rank == 0:
    lg = RectangleGenerator(particle_spacing_x1=0.1,
                            particle_spacing_x2=0.1)
    x, y, z = lg.get_coords()

    parray.add_property({'name':'x', 'data':x})
    parray.add_property({'name':'y', 'data':y})
    parray.add_property({'name':'z', 'data':z})
    parray.add_property({'name':'h'})
    parray.align_particles()
    parray.h[:] = 0.1
else:
    parray.add_property({'name':'x'})
    parray.add_property({'name':'y'})
    parray.add_property({'name':'z'})
    parray.add_property({'name':'h'})

pcm.add_array_to_bin(parray)
pcm.initialize()
