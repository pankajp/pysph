#!/bin/env python
"""
Simple test for checking if the control tree is setup properly.

Run this script with the following command

mpiexec -n [num_procs] python controller_check.py
"""

# logging setup
import logging
logging.basicConfig(level=logging.DEBUG, filename='/tmp/log_pysph', filemode='a')
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

from mpi4py import MPI
from pysph.parallel.parallel_controller import ParallelController

comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

logger.info('(%d)================controller_check====================='%(rank))

p = ParallelController(solver=None, cell_manager=None)


assert p.rank == rank
    
if rank == 0:
    assert p.parent_rank == -1
else:
    if rank % 2 == 0:
        assert p.parent_rank == ((rank)/2 -1)
    else:
        assert p.parent_rank == ((rank-1)/2)
        
if num_procs <= 2*rank + 1:
    assert p.l_child_rank == -1
    assert p.r_child_rank == -1
elif num_procs <= 2*rank + 2:
    assert p.l_child_rank == 2*rank + 1
    assert p.r_child_rank == -1
else:
    assert p.l_child_rank == 2*rank + 1
    assert p.r_child_rank == 2*rank + 2

logger.info('(%d)================controller_check====================='%(rank))


