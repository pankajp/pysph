""" Tests for the parallel cell manager """

import pysph.base.api as base
import pysph.parallel.api as parallel

import numpy
import pylab

# mpi imports
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
pid = comm.Get_rank()

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

# create the particle arrays on each processor
xc = numpy.linspace(0,8,9)
yc = numpy.linspace(0,8,9)
x,y = numpy.meshgrid(xc,yc)
x = x.ravel(); y = y.ravel()

dx = xc[1] - xc[0]
h = numpy.ones_like(x) * (dx - 1e-10)

name = "rank" + str(pid)
pa = base.get_particle_array(name=name, x=x, y=y, h=h)

if pid == 1:
    pa.x += 9

if pid == 2:
    pa.y += 9

if pid == 3:
    pa.x += 9; pa.y += 9

# create the cell manager
cm = parallel.ParallelCellManager(arrays_to_bin=[pa,], max_radius_scale=0.5,
                                  initialize=True, load_balancing=False,
                                  dimension=2)

cells_dict = cm.cells_dict
proc_map = cm.proc_map

# check for the number of cells on processor 0
if pid == 0:
    
    for id, cell in cells_dict.iteritems():
        draw_cell(cell)
        draw_particles(cell)

    for block_id in proc_map.local_block_map.keys():
        draw_block(proc_map.origin, proc_map.block_size, block_id)
        
    pylab.title("Cell structure on processor 0")
    pylab.show()

    print proc_map.local_block_map

if pid == 1:
    
    for id, cell in cells_dict.iteritems():
        draw_cell(cell)
        draw_particles(cell)

    for block_id in proc_map.local_block_map.keys():
        draw_block(proc_map.origin, proc_map.block_size, block_id)
        
    pylab.title("Cell structure on processor 1")
    pylab.show()

    print proc_map.local_block_map
