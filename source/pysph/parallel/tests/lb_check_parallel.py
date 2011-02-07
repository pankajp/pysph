""" Tests for the parallel cell manager """

import pysph.base.api as base
import pysph.parallel.api as parallel

from time import time

import numpy
import pylab

import pdb

# mpi imports
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
pid = comm.Get_rank()

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


xc = numpy.arange(0,1.0, 0.2)
x, y = numpy.meshgrid(xc,xc)

x = x = x.ravel()
y = y = y.ravel()
h = h = numpy.ones_like(x) * 0.25

dx = dy = 0.2
dx = dx

block_size = 0.5
cell_size = 0.5

block_000_indices = 0,1,2,5,6,7,10,11,12
block_100_indices = 3,4,8,9,13,14
block_010_indices = 15,16,17,20,21,22
block_110_indices = 18,19,23,24

name = "rank" + str(pid)

pa = base.get_particle_array(name="test", x=x, y=y, h=h)

pa.x += 1.0*pid
pa.x += 1e-10

pa.y += 1.0*(pid%2)
pa.y += 1e-10

# create the cell manager
cm = parallel.ParallelCellManager(arrays_to_bin=[pa,],
                                  max_radius_scale=2.0,
                                  dimension=2.0,
                                  load_balancing=False,
                                  initialize=False,
                                  min_cell_size=0.5)
t = time()
cm.initialize()
t = time() - t
print 'initialize time', t

cells_dict = cm.cells_dict
proc_map = cm.proc_map

print 'cells_dict'
print cells_dict
print
print 'block_map'
print proc_map.block_map

print 'load_per_proc'
print proc_map.load_per_proc

t = time()
cm.cells_update()
t = time() - t
print 'cells_update time', t

print 'cells_dict'
print cells_dict
print
print 'block_map'
print proc_map.block_map

print 'load_per_proc'
print proc_map.load_per_proc

print 'moving all but one blocks to proc 0'

t = time()
#send all but one block to proc=0
if pid > 0:
    cm.transfer_blocks_to_procs({0:proc_map.local_block_map.keys()[1:]},
                                recv_procs=[])
else:
    cm.transfer_blocks_to_procs({}, recv_procs=range(1,num_procs))
t = time() - t
print 'transfer_blocks time', t

t = time()
cm.cells_update()
t = time() - t
print 'cells_update time', t


print 'cells_dict'
print cells_dict
print
print 'block_map'
print proc_map.block_map

print 'load_per_proc'
print proc_map.load_per_proc

t = time()
cm.load_balancer.load_balance()
t = time() - t
print 'load_balance time', t



t = time()
#send all blocks to proc=0
if pid > 0:
    cm.transfer_blocks_to_procs({0:proc_map.local_block_map.keys()},
                                recv_procs=[])
else:
    cm.transfer_blocks_to_procs({}, recv_procs=range(1,num_procs))
t = time() - t
print 'transfer_blocks time', t

t = time()
cm.cells_update()
t = time() - t
print 'cells_update time', t


print 'cells_dict'
print cells_dict
print
print 'block_map'
print proc_map.block_map

print 'load_per_proc'
print proc_map.load_per_proc

print 'testing load_balance'

t = time()
cm.load_balancer.load_balance()
t = time() - t
print 'load_balance time', t

