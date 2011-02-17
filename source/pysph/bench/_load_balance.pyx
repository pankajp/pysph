"""
Module to get timings and performance metrics of the load balancer
"""

# MPI imports
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

from time import time

import sys
from sys import argv
flush = sys.stdout.flush
import os
from os.path import join, exists

from optparse import OptionParser

# logging imports
import logging

from pysph.sph.sph_calc import SPHBase
from pysph.sph.funcs.density_funcs import SPHRho

from load_balance_helper import parse_options, create_cell_manager, \
        get_lb_args, get_desc_name


def lb_test(args, lbargs=None):
    """returns a dictionary of parallel code timings"""
    options = parse_options(args)
    cm = create_cell_manager(options)
    #block, = cm.arrays_to_bin
    logger = options.logger
    
    processing_times = []
    communication_times = []
    total_iteration_times = []
    particle_counts = []
    
    # perform load balancing
    lb = cm.load_balancer
    lb.lb_max_iterations = options.num_load_balance_iterations
    lb.setup()
    if lbargs is None:
        get_lb_args()[0]
    t = time()
    print lbargs
    sys.stdout.flush()
    lb.load_balance(**lbargs)
    sys.stdout.flush()
    t = time() - t
    print rank, cm.get_number_of_particles(), 'real particles'
    cm.exchange_neighbor_particles()
    print rank, cm.get_number_of_particles(), 'real+dummy particles'
    print rank, len(cm.cells_dict), 'cells'
    
    print 'load balance time', t
    
    proc_blocks = comm.gather(cm.proc_map.local_block_map.keys(), root=0)
    
    np_per_proc = sum([parray.num_real_particles for parray in cm.arrays_to_bin])
    np_per_proc = comm.gather(np_per_proc, root=0)
    
    if rank == 0:
        method = get_desc_name(lbargs)
        block_proc = lb.get_block_proc(proc_blocks)
        metric = lb.get_metric(block_proc, np_per_proc)
        print 'metric :', metric
        lb.plot(proc_blocks, show=False, save_filename='lb_%s.png'%method)
    logger.info('load balancing over')
    
    # write the three times into a file.
    fname = os.path.join(options.destdir, 'stats_' + str(rank))
    file = open(fname, 'w')
    for i in range(len(total_iteration_times)):
        ln = str(total_iteration_times[i]) + ' '
        ln += str(communication_times[i]) + ' '
        ln += str(processing_times[i]) + '\n'
        file.write(ln)
    file.close()
    
    # write the particle counts
    fname = os.path.join(options.destdir, 'pcount_' + str(rank))
    file = open(fname, 'w')
    for i in range(len(total_iteration_times)):
        file.write(str(particle_counts[i]))
        file.write('\n')
    file.close()

    # write the VTK file if needed.
    if options.write_vtk:
        options.vtk_writer.write()

    if rank == 0:
        return {'time':t,
                'imbalance':metric[0],
                'cells_nbr':metric[1],
                'cells_nbr_proc':metric[2],
                'procs_nbr':metric[3],
                }

import functools

funcs = [functools.partial(lb_test,lbargs=i) for i in get_lb_args()]

def bench(args=None):
    """return a list of a dictionary of parallel benchmark timings"""
    if args is None:
        args = []
    timings = []
    
    for func in funcs:
        timings.append(func(args))
    return timings
    
if __name__ == '__main__':
    timings = bench(sys.argv[1:])
    print '(R=%d)'%(rank), timings

