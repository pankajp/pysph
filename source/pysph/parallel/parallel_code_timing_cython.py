#!/usr/bin/env python
"""
Module to get various timings of the parallel code.

What to do ?
 
    - Create a 2d square of particles.
    - Run SPHRho3D on it and find the time it takes for each iteration. 
"""

# MPI imports
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


import sys
from sys import argv
from optparse import OptionParser
op = OptionParser()
op.add_option('-w', '--width', dest='square_width',
              metavar='SQUARE_WIDTH')
op.add_option('-s', '--spacing', dest='particle_spacing',
              metavar='PARTICLE_SPACING')
op.add_option('-r', '--radius', dest='particle_radius',
              metavar='PARTICLE_RADIUS')
op.add_option('-d', '--destdir', dest='destdir',
              metavar='DESTDIR')
op.add_option('-i', '--sph-interpolations', dest='sph_interpolations',
              metavar='SPH_INTERPOLATIONS')
op.add_option('-n', '--num-iterations', dest='num_iterations',
              metavar='NUM_ITERATIONS')
op.add_option('-l', '--num-load-balance-iterations',
              dest='num_load_balance_iterations',
              metavar='NUM_LOAD_BALANCE_ITERATIONS')

destdir = '.'
square_width = 1.0
particle_spacing = 0.01
particle_radius = 0.01
num_iterations = 10
sph_interpolations = 1
num_load_balance_iterations = 100

# parse the input arguements
args = op.parse_args()
options = args[0]

import os
from os.path import join, exists

# setup the default values or the ones passed from the command line
if options.destdir is None:
    print 'No destination directory specified. Using current dir'
else:
    home = os.environ['HOME']
    destdir = home + '/'+ options.destdir+'/'

# create the destination directory if it does not exist.
if not exists(destdir):
    try:
        os.mkdir(destdir)
    except OSError:
	print 'Directory %s already exists'%(destdir)

# logging imports
import logging
logger = logging.getLogger()
log_filename = destdir + '/' + 'log_pysph_' + str(rank)
logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w')
logger.addHandler(logging.StreamHandler())

# read the square_width to use
if options.square_width == None:
    logger.warn('Using default square width of %f'%(square_width))
else:
    square_width = float(options.square_width)

# read the particle spacing
if options.particle_spacing == None:
    logger.warn('Using default particle spacing of %f'%(particle_spacing))
else:
    particle_spacing = float(options.particle_spacing)

# read the particle radius
if options.particle_radius == None:
    logger.warn('Using default particle radius of %f'%(particle_radius))
else:
    particle_radius = float(options.particle_radius)

# read the number of sph-interpolations to perform
if options.sph_interpolations == None:
    logger.warn('Using default number of SPH interpolations %f'%(
            sph_interpolations))
else:
    sph_interpolations = int(options.sph_interpolations)

# read the total number of iterations to run
if options.num_iterations == None:
    logger.warn('Using default number of iterations %d'%(num_iterations))
else:
    num_iterations = int(options.num_iterations)
if options.num_load_balance_iterations == None:
    logger.warn('Running 100 initial load balance iterations')
else:
    num_load_balance_iterations = int(options.num_load_balance_iterations)

# one node zero - write this setting into a file.
if rank == 0:
    settings_file = destdir + '/settings.dat'
    f = open(settings_file, 'w')
    f.write('Run with command : %s\n'%(sys.argv))
    f.write('destdir = %s\n'%(destdir))
    f.write('square_width = %f\n'%(square_width))
    f.write('particle_spacing = %f\n'%(particle_spacing))
    f.write('particle_radius = %f\n'%(particle_radius))
    f.write('sph_interpolations = %d\n'%(sph_interpolations))
    f.write('num_iterations = %d\n'%(num_iterations))
    f.close()


# local imports
from pysph.base.kernel2d import CubicSpline2D
from pysph.base.point import Point
from pysph.base.nnps import *

from pysph.parallel.cy_parallel_cell import *
from pysph.parallel.parallel_controller import ParallelController
from pysph.parallel.parallel_component import ParallelComponent
from pysph.solver.fluid import Fluid
from pysph.solver.solver_base import *
from pysph.parallel.cy_dummy_solver import *
from pysph.solver.particle_generator import DensityComputationMode as Dcm
from pysph.solver.particle_generator import MassComputationMode as Mcm
from pysph.solver.basic_generators import RectangleGenerator
from pysph.solver.parallel_vtk_writer import ParallelVTKWriter
from pysph.sph.sph_calc import SPHBase
from pysph.sph.density_funcs import SPHRho3D
from pysph.solver.vtk_writer import VTKWriter

# create a parallel controller.
controller = ParallelController()

# create a component manager.
component_manager = ComponentManager()
parallel_component = ParallelComponent(name='parallel_component', solver=None)
component_manager.add_component(parallel_component)

# create a parallel cell manager.
cell_manager = ParallelCellManager(parallel_controller=controller,
                                   initialize=False,
                                   solver=None)
# enable load balancing
cell_manager.load_balancing = False
cell_manager.load_balancer.skip_iteration = 1
cell_manager.load_balancer.threshold_ratio = 10.
cell_manager.dimension = 2

# create a nnps manager
nnps_manager = NNPSManager(cell_manager=cell_manager)

# create a dummy solver - serves no purpose but for a place holder
solver = DummySolver(cell_manager=cell_manager,
                     nnps_manager=nnps_manager,
                     parallel_controller=controller)
cell_manager.solver=solver
parallel_component.solver = solver
cell_manager.load_balancer.solver = solver

# the kernel to be used
kernel = CubicSpline2D()

# create the square block of particles.
start_point = Point(0, 0, 0)
end_point = Point(square_width, square_width, 0)

block = Fluid(name='block')
solver.add_entity(block)
parray = block.get_particle_array()
if rank == 0:
    rg = RectangleGenerator(start_point=start_point,
                            end_point=end_point,
                            particle_spacing_x1=particle_spacing,
                            particle_spacing_x2=particle_spacing,
                            density_computation_mode=Dcm.Set_Constant,
                            particle_density=1000.0,
                            mass_computation_mode=Mcm.Compute_From_Density,
                            particle_h=particle_radius,
                            kernel=kernel,
                            filled=True)
    block.add_particles(rg.get_particles())

# create a parallel vtk writer
data_file_name = 'block_data_' + str(rank)
vtk_writer = ParallelVTKWriter(solver=solver, 
                               entity_list=[block],
                               file_name_prefix=data_file_name,
                               scalars=['rho', 'pid', '_tmp', 'h', 'm', 'tag'],
                               vectors={})
component_manager.add_component(vtk_writer)

solver._setup_entities()

if rank != 0:
    # add some necessary properties to the particle array.
    parray.add_property({'name':'x'})
    parray.add_property({'name':'y'})
    parray.add_property({'name':'z'})
    parray.add_property({'name':'h', 'default':particle_radius})
    parray.add_property({'name':'rho', 'default':1000.})
    parray.add_property({'name':'pid'})
    parray.add_property({'name':'_tmp', 'default':0.0})
    parray.add_property({'name':'m'})
else:
    parray.add_property({'name':'_tmp'})
    parray.add_property({'name':'pid', 'default':0.0})

solver._setup_nnps()

# create the SPHRho3D class
parray = block.get_particle_array()

# create sph_interpolations number of sph_funcs
sph_funcs = [None]*sph_interpolations
sph_sums = [None]*sph_interpolations
for i in range(sph_interpolations):
    sph_funcs[i] = SPHRho3D(source=parray, dest=parray)
    sph_sums[i] = SPHBase(sources=[parray], dest=parray, kernel=kernel,
                          sph_funcs=[sph_funcs[i]],
                          nnps_manager=nnps_manager)

# sph_func = SPHRho3D(source=parray, dest=parray)
# sph_sum = SPHBase(sources=[parray], dest=parray, kernel=kernel,
#                   sph_funcs=[sph_func],
#                   nnps_manager=nnps_manager)
vtk_writer.setup_component()

processing_times = []
communication_times = []
total_iteration_times = []
particle_counts = []

import time

particle_counts.append(parray.get_number_of_particles())

# perform load balancing before actual iterations begin.
lb = cell_manager.load_balancer
lb.lb_max_iterations = num_load_balance_iterations
lb.setup()
lb.load_balance()
cell_manager.root_cell.exchange_neighbor_particles()

for i in range(num_iterations):

    t1 = time.time()

    # parallel operations.
    #vtk_writer.write()
    nnps_manager.py_update()
    t2 = time.time()
    
    communication_times.append(t2-t1)
    parr = block.get_particle_array()
    t2 = time.time()

    particle_counts.append(parr.num_real_particles)

    t2 = time.time()
    # computation
    for j in range(sph_interpolations):
        sph_sums[j].sph1('_tmp')
        parr.rho[:] = parr._tmp

    t3 = time.time()

    processing_times.append(t3-t2)

    parray.set_dirty(True)
    solver.current_iteration += 1

    t4 = time.time()

    total_iteration_times.append(t4-t1)
    logger.info('Iteration %d done'%(i))

# write the three times into a file.
fname = destdir + '/' + 'stats_' + str(rank)

file = open(fname, 'w')

for i in range(len(total_iteration_times)):
    ln = str(total_iteration_times[i]) + ' '
    ln += str(communication_times[i]) + ' '
    ln += str(processing_times[i]) + '\n'
    file.write(ln)

file.close()

# write the particle counts
fname = destdir + '/' + 'pcount_' + str(rank)

file = open(fname, 'w')

for i in range(len(total_iteration_times)):
    file.write(str(particle_counts[i]))
    file.write('\n')

file.close()
