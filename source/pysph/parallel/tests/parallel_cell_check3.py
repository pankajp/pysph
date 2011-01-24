""" Tests for the parallel cell manager """

import pysph.base.api as base
import pysph.parallel.api as parallel

import numpy
import pylab
import time

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

def draw_cell(cell, color="b"):
    centroid = base.Point()
    cell.get_centroid(centroid)
    
    half_size = 0.5 * cell.cell_size

    x1, y1 = centroid.x - half_size, centroid.y - half_size
    x2, y2 = x1 + cell.cell_size, y1
    x3, y3 = x2, y1 + cell.cell_size
    x4, y4 = x1, y3

    pylab.plot([x1,x2,x3,x4,x1], [y1, y2, y3, y4,y1], color)

def draw_block(origin, block_size, block_id, color="r"):

    half_size = 0.5 * block_size
    x,y = [], []

    xc = origin.x + ((block_id.x + 0.5) * proc_map.block_size)        
    yc = origin.y + ((block_id.y + 0.5) * proc_map.block_size)
        
    x1, y1 = xc - half_size, yc - half_size
    x2, y2 = x1 + block_size, y1
    x3, y3 = x2, y2 + block_size
    x4, y4 = x1, y3
    
    pylab.plot([x1,x2,x3,x4,x1], [y1, y2, y3, y4,y1], color)

def draw_particles(cell, color="y"):
    arrays = cell.arrays_to_bin
    num_arrays = len(arrays)
    
    index_lists = []
    cell.get_particle_ids(index_lists)

    x, y = [], []

    for i in range(num_arrays):
        array = arrays[i]
        index_array = index_lists[i]
        
        indices = index_lists[i].get_npy_array()

        xarray, yarray = array.get('x','y')
        for j in indices:
            x.append(xarray[j])
            y.append(yarray[j])

    pylab.plot(x,y,color+"o")

def get_sorted_indices(cell):
    index_lists = []
    cell.get_particle_ids(index_lists)
    index_array = index_lists[0].get_npy_array()
    index_array.sort()

    print type(index_array)
    return index_array

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
pa = pa = base.get_particle_array(name="test", x=x, y=y, h=h)

if pid == 1:
    pa.x += 1.0
    pa.x += 1e-10

if pid == 2:
    pa.y += 9

if pid == 3:
    pa.x += 9; pa.y += 9

# create the cell manager

cm = cm = parallel.ParallelCellManager(arrays_to_bin=[pa,],
                                       max_radius_scale=2.0,
                                       dimension=2.0,
                                       load_balancing=False,
                                       initialize=False,
                                       min_cell_size=0.5)

# find global min and max

cm.update_global_properties()

# setup the origin

cm.setup_origin()

# compute block size

cm.compute_block_size(0.5)

# compute cell size

cm.compute_cell_size(0,0)

# setup array indices.

cm.py_rebuild_array_indices()

# setup the cells_dict

cm.py_setup_cells_dict()
        
# setup information for the processor map.

cm.setup_processor_map()

# build a single cell with all the particles

cm._build_cell()


cells_dict = cm.cells_dict
proc_map = cm.proc_map


# Test the initial setup

assert len(cells_dict) == 1, "At this stage only the base cell should exist"

cell = cells_dict.values()[0]

index_lists = []
cell.get_particle_ids(index_lists)
index_array = index_lists[0].get_npy_array()
index_array.sort()

# check the indices

cid = cells_dict.keys()[0]

assert (cid.x, cid.y, cid.z) == [(0,0,0),(2,0,0)][pid]

for i in range(25):
    assert index_array[i] == i       

# test origin and the block size for the processor map

origin = proc_map.origin
assert (origin.x, origin.y, origin.z) == (0,0,0)

assert proc_map.block_size == 0.5

print "Checking cells_update"

# check bin_particles

print "Testing bin particles: new_block_cells, remote_block_cells"

new_block_cells, remote_block_cells = cm.bin_particles()

# the local and global proc_map should be empty

assert len(proc_map.local_block_map) == 0
assert len(proc_map.block_map) == 0

# the remote block cells should be empty

assert len(remote_block_cells) == 0

# there should be four new block cells

bid1 = base.IntPoint(0+2*pid,0,0)
bid2 = base.IntPoint(1+2*pid,0,0)
bid3 = base.IntPoint(1+2*pid,1,0)
bid4 = base.IntPoint(0+2*pid,1,0)

print new_block_cells

assert new_block_cells.has_key(bid1)
assert new_block_cells.has_key(bid2)
assert new_block_cells.has_key(bid3)
assert new_block_cells.has_key(bid4)

# the cells dict should be empty as well at this point

assert len(cells_dict) == 0

# test the particle copies for the new blocks

print "Testing create_new_particle_copies"

new_block_particles = cm.create_new_particle_copies(new_block_cells,
                                                    False)

assert len(new_block_particles) == 4

# check particles in bid 0,0,0

parray_list = new_block_particles.get(bid1)
assert len(parray_list) == 1

parray = parray_list[0]

indices = parray.get("idx")
indices.sort()
    
assert list(indices) == list(block_000_indices)

# check particles in bid 1,0,0

parray_list = new_block_particles.get(bid2)
assert len(parray_list) == 1

parray = parray_list[0]

indices = parray.get("idx")
indices.sort()
    
assert list(indices) == list(block_100_indices)

# check particles in bid 1,1,0

parray_list = new_block_particles.get(bid3)
assert len(parray_list) == 1

parray = parray_list[0]

indices = parray.get("idx")
indices.sort()
    
assert list(indices) == list(block_110_indices)

# check particles in bid 0,1,0

parray_list = new_block_particles.get(bid4)
assert len(parray_list) == 1

parray = parray_list[0]

indices = parray.get("idx")
indices.sort()
    
assert list(indices) == list(block_010_indices)

print "Testing assign_new_blocks: proc_map"

# assign the new blocks to the processor map

cm.assign_new_blocks(new_block_cells)

# check the processor map

assert len(cm.proc_map.local_block_map) == 4
assert len(cm.proc_map.block_map) == 4
assert cm.proc_map.nbr_procs == [pid]

# compute cell size 

cm.compute_cell_size()

assert cm.cell_size == 0.5

# ensure all particles are local (!=0)

pa = cm.arrays_to_bin[0]
local = pa.get("local", only_real_particles=False)
for i in range(pa.get_number_of_particles()):
    assert local[i] != 0

print "Testing rebin particles"

# rebin particles

cm.rebin_particles()

# now check the cells_dict

cells_dict = cm.cells_dict

assert len(cells_dict) == 4

# check the particles in the cells

cids = [base.IntPoint(0+2*pid,0,0), base.IntPoint(1+2*pid,0,0), 
        base.IntPoint(1+2*pid,1,0), base.IntPoint(0+2*pid,1,0)]

index_map = [block_000_indices, block_100_indices,
             block_110_indices, block_010_indices]

for i in range(4):
    cid = cids[i]
    cell = cells_dict.get(cid)

    index_lists = []
    cell.get_particle_ids(index_lists)
    cell_indices = index_lists[0].get_npy_array()
    cell_indices.sort()

    assert list(cell_indices) == list(index_map[i])


print "Testing glb_update_proc_map"

# update the global processor map

cm.glb_update_proc_map()

# check the processor maps

print "Processor", pid, "Block Maps"

print "Local\n"
for blockid in cm.proc_map.local_block_map:
    print blockid
print

print "Global\n"
for blockid in cm.proc_map.block_map:
    print blockid, cm.proc_map.block_map[blockid]
print

print "Testing Neighbors", pid
assert cm.proc_map.nbr_procs == [0,1]

# exchange neighbor particles

cm.exchange_neighbor_particles()

print "Testing Exchange Neighbor Particles"

print "Cells Dict For Processor 0 After Exchange\n"
for cid, cell in cells_dict.iteritems():
    print cid, "np = ", cell.get_number_of_particles()

#print_yours=True
#comm.send(obj=print_yours, dest=1)

print "Testing remote particle indices on Processor 0"

parray = cm.arrays_to_bin[0]
np = parray.get_number_of_particles()
nrp = parray.num_real_particles

assert nrp == 25
assert np == [40,35][pid]

local = parray.get("local", only_real_particles=False)

rpi = cm.remote_particle_indices[1-pid][0]

assert rpi[0] == nrp
assert rpi[1] == (np - 1)

for i in range(np):
    if i >= nrp:
        assert local[i] == 0
    else:
        assert local[i] == 1

# test the update of remote particle indices

print "Testing Update Remote Particle Properties on processor 0"

# change the local property say 'p' and 'rho' to -1

pa = cm.arrays_to_bin[0]

p = pa.get('p', only_real_particles=False)
rho = pa.get('rho', only_real_particles=False)

p[:nrp] = -1
rho[:nrp] = -1

for i in range(np):
    if i >= nrp:
        assert p[i] != -1
        assert rho[i] != -1

#yours_is_set = comm.recv(source=1)

#if yours_is_set:
cm.update_remote_particle_properties([['p','rho']])

p = pa.get('p', only_real_particles=False)
rho = pa.get('rho', only_real_particles=False)

for i in range(np):
    if i >= nrp:
        assert p[i] == -1
        assert rho[i] == -1


#####################################################################
# SECOND ITERATION
#####################################################################

# test the configuration 

cids = [base.IntPoint(0+pid,0,0), base.IntPoint(1+pid,0,0), base.IntPoint(1+pid,1,0),
        base.IntPoint(0+pid,1,0), base.IntPoint(2+pid,0,0), base.IntPoint(2+pid,1,0)]

pa = cm.arrays_to_bin[0]
for cid in cids:
    assert cm.cells_dict.has_key(cid)
    if cid in [base.IntPoint(2-pid,0,0), base.IntPoint(2-pid,1,0)]:
        cell = cells_dict.get(cid)
        index_lists = []
        cell.get_particle_ids(index_lists)            
        parray = pa.extract_particles(index_lists[0])
        local =  parray.get('local', only_real_particles=False)
        for val in local:
            assert val == 0

# remove non local particles

cm.remove_remote_particles()
cm.delete_empty_cells()

np = pa.get_number_of_particles()

assert np == 25

# move 6 particles in cell/block (1,0,0) to (2,0,0) in pid=0

x = pa.get('x')

if pid == 0:
    for i in block_100_indices:
        x[i] += 0.5


print "Local\n"
for blockid in cm.proc_map.local_block_map:
    print blockid
print

print "Global\n"
for blockid in cm.proc_map.block_map:
    print blockid, cm.proc_map.block_map[blockid]
print

if True:
    cm.cells_update()
else:
    
    cm.remove_remote_particles()
    cm.delete_empty_cells()
    print 'cells_dict'
    print cm.cells_dict.values()

    # wait till all processors have reached this point.

    cm.parallel_controller.comm.Barrier()

    logger.debug('++++++++++++++++ UPDATE BEGIN +++++++++++++++++++++')

    # bin the particles and find the new_cells and remote_cells.

    new_block_cells, remote_block_cells = cm.bin_particles()
    #cm.delete_empty_cells()
    # create particle copies and mark those particles as remote.

    #cm.new_particles_for_neighbors = cm.create_new_particle_copies(
    #    remote_block_cells, True)

    # exchange particles moving into another processor

    #cm.exchange_crossing_particles_with_neighbors(
    #                            cm.new_particles_for_neighbors)

    # remove any remote particles
    print 'binned_particles'
    print remote_block_cells
    print new_block_cells
    print cm.cells_dict
    
    # assign crossing particles' blocks 
    cm.mark_crossing_particles(remote_block_cells)
    new_particles_for_neighbors = cm.create_new_particle_copies(
        remote_block_cells, True)
    # assign new blocks based on the new_block_cells
    cm.assign_new_blocks(new_block_cells)
    new_region_particles = cm.create_new_particle_copies(
        new_block_cells, True)
    trf_particles = new_particles_for_neighbors
    trf_particles.update(new_region_particles)
    
    npr = sum([i.num_real_particles for i in cm.arrays_to_bin])
    assert comm.bcast(comm.reduce(npr)) == 50
    
    npr = sum([i.num_real_particles for i in cm.arrays_to_bin])
    nprt = comm.bcast(comm.reduce(npr))
    assert nprt==50, 'num_particles = %d != 50, %d'%(nprt,npr)
    
    cm.remove_remote_particles()
    # compute the cell sizes for binning

    cm.compute_cell_size()

    # rebin the particles

    cm.rebin_particles()

    # wait till all processors have reached this point.

    cm.parallel_controller.comm.Barrier()
    
    # update the processor map and resolve the conflicts
    #pdb.set_trace()
    if False:
        cm.glb_update_proc_map()
    else:
        block_particles = {}
        cm.proc_map.update()
        print 'block_map'
        print cm.proc_map.block_map
        
        TAG_PROC_MAP_UPDATE = 101
        # merge data from all children proc maps.
        for c_rank in cm.pc.children_proc_ranks:
            c_proc_map = comm.recv(source=c_rank,
                                   tag=TAG_PROC_MAP_UPDATE)
            cm.proc_map.merge(c_proc_map)
        
        # we now have partially merged data, send it to parent if not root.
        if cm.pc.parent_rank > -1:
            comm.send(cm.proc_map, dest=cm.pc.parent_rank,
                      tag=TAG_PROC_MAP_UPDATE)

            # receive updated proc map from parent
            updated_proc_map = comm.recv(source=cm.pc.parent_rank,
                                         tag=TAG_PROC_MAP_UPDATE)

            # set our proc data with the updated data.
            cm.proc_map.block_map.clear()
            cm.proc_map.block_map.update(updated_proc_map.block_map)
            
            cm.proc_map.conflicts.clear()
            cm.proc_map.conflicts.update(updated_proc_map.conflicts)

        # send updated data to children.
        for c_rank in cm.pc.children_proc_ranks:
            comm.send(cm.proc_map, dest=c_rank, tag=TAG_PROC_MAP_UPDATE)
        
        # now all procs have same proc_map
        # resolve any conflicts
        print 'conflicts'
        print cm.proc_map.conflicts
        if cm.proc_map.conflicts:
            # calculate num_blocks per proc
            blocks_per_proc = {}
            recv_procs = set([cm.pid])
            procs_blocks_particles = {cm.pid:{}}
            
            for bid in cm.proc_map.block_map:
                proc = cm.proc_map.block_map[bid]
                blocks_per_proc[proc] = blocks_per_proc.get(proc, 0)
            
            #pdb.set_trace()
            # assign block to proc with least blocks then max rank
            for bid in cm.proc_map.conflicts:
                proc = cm.proc_map.block_map.get(bid)
                candidates = list(cm.proc_map.conflicts[bid])
                if proc < 0:
                    proc = candidates[0]
                    blocks = blocks_per_proc[0]
                    
                    # find the winning proc for each block
                    for i in range(1, len(candidates)):
                        if blocks_per_proc[i] < blocks:
                            proc = candidates[i]
                            blocks = blocks_per_proc[i]
                        elif blocks_per_proc[i] == blocks:
                            if candidates[i] > proc:
                                proc = candidates[i]
                    
                    cm.proc_map.block_map[bid] = proc
                    blocks_per_proc[proc] += 1
                
                if proc != cm.pid:
                    if proc in procs_blocks_particles:
                        procs_blocks_particles[proc][bid] = trf_particles[bid]
                    else:
                        procs_blocks_particles[proc] = {bid:trf_particles[bid]}
                    # send to winning proc
                    if bid in cm.proc_map.local_block_map:
                        del cm.proc_map.local_block_map[bid]
                else:
                    # recv from other conflicting procs
                    cm.proc_map.local_block_map[bid] = proc
                    recv_procs.update(candidates)
            
            logger.info('remote_block_particles: '+str(procs_blocks_particles))
            #if self.pid in recv_procs: recv_procs.remove(self.pid)
            cm.transfer_particles_to_procs(procs_blocks_particles,
                                          recv_procs=list(recv_procs))
            # remove the transferred particles
            cm.remove_remote_particles()
            #cm.delete_empty_cells()
            cm.proc_map.conflicts.clear()
        
        # setup the region neighbors.
        cm.proc_map.find_region_neighbors()

    # call a load balancer function.

    if cm.initialized == True:
        if cm.load_balancing == True:
            cm.load_balancer.load_balance()

    logger.info('cells_update:'+str([parr.get_number_of_particles()
                                        for parr in cm.arrays_to_bin]))

    # exchange neighbor information

    cm.exchange_neighbor_particles()

npr = sum([i.num_real_particles for i in cm.arrays_to_bin])
nprt = comm.bcast(comm.reduce(npr))
assert nprt==50, 'num_particles = %d != 50, %d'%(nprt,npr)

print "Local\n"
for blockid in cm.proc_map.local_block_map:
    print blockid
print

print "Global\n"
for blockid in cm.proc_map.block_map:
    print blockid, cm.proc_map.block_map[blockid]
print

print cm.cells_dict.values()

np = pa.get_number_of_particles()
nrp = pa.num_real_particles
print pid, np, nrp
assert np == [40,35][pid], '%r, %r'%(pid, np)
assert nrp == [19,31][pid], '%r, %r'%(pid, np)

# now move the 4 particles in cell/block (1,1,0) to block/cell (1,2,0) in pid=0
# and particles in cell (2,1,0) to cell (1, 2, 0) in pid=1

x, y = pa.get('x', 'y')

cell = cm.cells_dict.get(base.IntPoint([1,2][pid],1,0))
index_lists = []
cell.get_particle_ids(index_lists)
index_array = index_lists[0].get_npy_array()
print index_array
for i in index_array:
    y[i] += 0.5
    if pid == 1:
        x[i] -= 0.5

# now call a cells update
cm.cells_update()

print "Local\n"
for blockid in cm.proc_map.local_block_map:
    print blockid
print

print "Global\n"
for blockid in cm.proc_map.block_map:
    print blockid, cm.proc_map.block_map[blockid]
print

print cm.cells_dict.values()

npr = sum([i.num_real_particles for i in cm.arrays_to_bin])
nprt = comm.bcast(comm.reduce(npr))
assert nprt==50, 'num_particles = %d != 50, %d'%(nprt,npr)

np = pa.get_number_of_particles()
nrp = pa.num_real_particles

print pid, np

assert nrp == [19 + 6, 31 - 6][pid]
#assert np  == nrp + 10
#assert np == [np, 41][pid]
assert np == nrp


cells_nps = {base.IntPoint(0,0,0):9,
             base.IntPoint(2,0,0):15,
             base.IntPoint(3,0,0):6,
             base.IntPoint(0,1,0):6,
             base.IntPoint(3,1,0):4,
             base.IntPoint(1,2,0):10,
             }

print cm.proc_map.block_map
for cid, cell in cm.cells_dict.iteritems():
    print cid, cell, cell.get_number_of_particles()
    assert cell.get_number_of_particles() == cells_nps[cid], '%r'%(cell)

npr = sum([i.num_real_particles for i in cm.arrays_to_bin])
assert comm.bcast(comm.reduce(npr)) == 50
assert nrp == [19 + 6, 31 - 6][pid]

