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

from time import time

import sys
from sys import argv
import os
from os.path import join, exists

from optparse import OptionParser

# logging imports
import logging

# local imports
from pysph.base.kernels import CubicSplineKernel
from pysph.base.point import Point
from pysph.base.nnps import NNPSManager
from pysph.parallel.parallel_cell import ParallelCellManager
from pysph.parallel.parallel_controller import ParallelController
from pysph.parallel.parallel_component import ParallelComponent
from pysph.base.particle_array cimport ParticleArray
from pysph.solver.fluid import Fluid
from pysph.solver.solver_base import SolverBase, ComponentManager
from pysph.parallel.dummy_solver import DummySolver
from pysph.solver.particle_generator import DensityComputationMode as Dcm
from pysph.solver.particle_generator import MassComputationMode as Mcm
from pysph.solver.basic_generators import RectangleGenerator
from pysph.solver.parallel_vtk_writer import ParallelVTKWriter
from pysph.sph.sph_calc import SPHBase
from pysph.sph.density_funcs import SPHRho3D
from pysph.solver.vtk_writer import VTKWriter

class ParallelCellManagerTest(ParallelCellManager):
    """A parallel cell manager for timings code"""
    def __init__(self, arrays_to_bin=[],
                 min_cell_size=0.1, 
                 max_cell_size=0.5,
                 origin_point=Point(0, 0, 0),
                 num_levels=2,
                 initialize=True,
                 parallel_controller=None,
                 max_radius_scale=2.0,
                 dimension=3,
                 load_balancing=True,
                 solver=None,
                 max_cell_scale=2.0,
                 *args, 
                 **kwargs):

        self.max_cell_scale = max_cell_scale
        
        ParallelCellManager.__init__(
            self, 
            arrays_to_bin=arrays_to_bin,
            min_cell_size=min_cell_size,
            max_cell_size=max_cell_size,
            origin_point=origin_point,
            num_levels=num_levels,
            initialize=initialize,
            parallel_controller=parallel_controller,
            max_radius_scale=max_radius_scale,
            dimension=dimension,
            load_balancing=load_balancing,
            solver=solver,
            *args, **kwargs)
        
    def setup_cell_sizes(self):
        """
        Sets up the cell sizes to use from the 'h' values.
        
        The smallest cell size is set to 2*max_radius_scale*min_h
        The larger cell size is set to max_cell_scale*smallest_cell_size
        
        Set the number of levels to 2.
        """
        self.min_cell_size = 2*self.max_radius_scale*self.glb_max_h
        self.max_cell_size = self.max_cell_scale*self.min_cell_size
        self.num_levels = 2
        pc = self.parallel_controller


def parse_options(args=None):
    """parse commandline options from given list (default=sys.argv[1:])"""
    # default values
    square_width = 1.0
    particle_spacing = 0.01
    particle_radius = 0.01
    sph_interpolations = 1
    num_iterations = 10
    num_load_balance_iterations = 100
    max_cell_scale = 2.0

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
    op.add_option('-o', '--write-vtk', 
                  action="store_true", default=False, dest='write_vtk',
                  help='write a vtk file after all iterations are done')
    op.add_option('-v', '--verbose',
                  action="store_true", default=False, dest='verbose',
                  help='print large amounts of debug information')
    op.add_option('-c', '--max-cell-scale', dest='max_cell_scale',
                  metavar='MAX_CELL_SCALE',
                  help='specify the ratio of largest cell to smallest cell')
    
    # parse the input arguements
    args = op.parse_args()
    options = args[0]

    # setup the default values or the ones passed from the command line
    if options.destdir is None:
        print 'No destination directory specified. Using current dir'
        options.destdir = ''
    options.destdir = os.path.abspath(options.destdir)

    # create the destination directory if it does not exist.
    if not exists(options.destdir):
        os.mkdir(options.destdir)

    # logging
    options.logger = logger = logging.getLogger()
    log_filename = os.path.join(options.destdir, 'log_pysph')
    if options.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(level=log_level, filename=log_filename, filemode='w')
    #logger.addHandler(logging.StreamHandler())

    # read the square_width to use
    if options.square_width == None:
        logger.warn('Using default square width of %f'%(square_width))
        options.square_width = square_width
    options.square_width = float(options.square_width)
    
    # read the particle spacing
    if options.particle_spacing == None:
        logger.warn('Using default particle spacing of %f'%(particle_spacing))
        options.particle_spacing = particle_spacing
    options.particle_spacing = float(options.particle_spacing)
    
    # read the particle radius
    if options.particle_radius == None:
        logger.warn('Using default particle radius of %f'%(particle_radius))
        options.particle_radius = particle_radius
    options.particle_radius = float(options.particle_radius)
    
    # read the number of sph-interpolations to perform
    if options.sph_interpolations == None:
        logger.warn('Using default number of SPH interpolations %f'%(
                sph_interpolations))
        options.sph_interpolations = sph_interpolations
    options.sph_interpolations = int(sph_interpolations)
    
    # read the total number of iterations to run
    if options.num_iterations == None:
        logger.warn('Using default number of iterations %d'%(num_iterations))
        options.num_iterations = num_iterations
    options.num_iterations = int(options.num_iterations)

    if options.num_load_balance_iterations == None:
        logger.warn('Running %d initial load balance iterations'
                    %(num_load_balance_iterations))
        options.num_load_balance_iterations = num_load_balance_iterations
    options.num_load_balance_iterations = int(num_load_balance_iterations)
    
    if options.max_cell_scale == None:
        logger.warn('Using default max cell scale of %f'%(max_cell_scale))
        options.max_cell_scale = max_cell_scale
    options.max_cell_scale = float(options.max_cell_scale)
    
    # one node zero - write this setting into a file.
    if rank == 0:
        settings_file = options.destdir + '/settings.dat'
        f = open(settings_file, 'w')
        f.write('Run with command : %s\n'%(sys.argv))
        f.write('destdir = %s\n'%(options.destdir))
        f.write('square_width = %f\n'%(options.square_width))
        f.write('particle_spacing = %f\n'%(options.particle_spacing))
        f.write('particle_radius = %f\n'%(options.particle_radius))
        f.write('sph_interpolations = %d\n'%(options.sph_interpolations))
        f.write('num_iterations = %d\n'%(options.num_iterations))
        f.close()

    return options

def setup_solver(options):
    """sets up and returns a parallel solver"""
    
    # create a parallel controller.
    controller = ParallelController()
    
    # create a component manager.
    component_manager = ComponentManager()
    parallel_component = ParallelComponent(name='parallel_component',
                                           solver=None)
    component_manager.add_component(parallel_component)
    
    # create a parallel cell manager.
    cell_manager = ParallelCellManagerTest(parallel_controller=controller,
                                       initialize=False,
                                       solver=None,
                                       max_cell_scale=options.max_cell_scale)
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
    solver.kernel = kernel = CubicSplineKernel()
    
    # create the square block of particles.
    start_point = Point(0, 0, 0)
    end_point = Point(options.square_width, options.square_width, 0)
    
    block = Fluid(name='block')
    solver.add_entity(block)
    parray = block.get_particle_array()
    if rank == 0:
        rg = RectangleGenerator(start_point=start_point,
                                end_point=end_point,
                                particle_spacing_x1=options.particle_spacing,
                                particle_spacing_x2=options.particle_spacing,
                                density_computation_mode=Dcm.Set_Constant,
                                particle_density=1000.0,
                                mass_computation_mode=Mcm.Compute_From_Density,
                                particle_h=options.particle_radius,
                                kernel=kernel,
                                filled=True)
        block.add_particles(rg.get_particles())
    
    # create a parallel vtk writer
    data_file_name = 'block_data_' + str(rank)
    vtk_writer = ParallelVTKWriter(solver=solver, 
                                   entity_list=[block],
                                   file_name_prefix=data_file_name,
                                   scalars=['rho','pid','_tmp','h','m','tag'],
                                   vectors={})
    component_manager.add_component(vtk_writer)
    
    solver._setup_entities()
    
    if rank != 0:
        # add some necessary properties to the particle array.
        parray.add_property({'name':'x'})
        parray.add_property({'name':'y'})
        parray.add_property({'name':'z'})
        parray.add_property({'name':'h', 'default':options.particle_radius})
        parray.add_property({'name':'rho', 'default':1000.})
        parray.add_property({'name':'pid'})
        parray.add_property({'name':'_tmp', 'default':0.0})
        parray.add_property({'name':'m'})
    else:
        parray.add_property({'name':'_tmp'})
        parray.add_property({'name':'pid', 'default':0.0})
    
    solver._setup_nnps()
    
    vtk_writer.setup_component()
    options.vtk_writer = vtk_writer
    
    return solver


def parallel_timings(args):
    """returns a dictionary of parallel code timings"""
    options = parse_options(args)
    solver = setup_solver(options)
    logger = options.logger
    
    block = solver.entity_list[0]
    parray = block.get_particle_array()
    
    # create sph_interpolations number of sph_funcs
    sph_funcs = [None]*options.sph_interpolations
    sph_sums = [None]*options.sph_interpolations
    
    # create the SPHRho3D class
    for i in range(options.sph_interpolations):
        sph_funcs[i] = SPHRho3D(source=parray, dest=parray)
        sph_sums[i] = SPHBase(sources=[parray], dest=parray,
                              kernel=solver.kernel,
                              sph_funcs=[sph_funcs[i]],
                              nnps_manager=solver.nnps_manager)
    
    processing_times = []
    communication_times = []
    total_iteration_times = []
    particle_counts = []
    
    particle_counts.append(parray.get_number_of_particles())
    
    # perform load balancing before actual iterations begin.
    lb = solver.cell_manager.load_balancer
    lb.lb_max_iterations = options.num_load_balance_iterations
    lb.setup()
    lb.load_balance()
    solver.cell_manager.root_cell.exchange_neighbor_particles()
    
    N = options.num_iterations
    for i in range(N):
    
        t1 = time()
    
        # parallel operations.
        solver.nnps_manager.py_update()
        t2 = time()
        
        communication_times.append(t2-t1)
        parr = block.get_particle_array()
        particle_counts.append(parr.num_real_particles)
    
        t2 = time()
        # computation
        for j in range(options.sph_interpolations):
            sph_sums[j].sph1('_tmp')
            parr.rho[:] = parr._tmp
    
        t3 = time()
        processing_times.append(t3-t2)
    
        parr.set_dirty(True)
        solver.current_iteration += 1
        t4 = time()
    
        total_iteration_times.append(t4-t1)
    
        logger.info('Iteration %d done'%(i))
    
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

    return {'processing_time':sum(processing_times)/N,
            'communication_time':sum(communication_times)/N,
            'total_iteration_time':sum(total_iteration_times)/N
            }


funcs = [parallel_timings]


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
    print timings
