""" some utility function for use in load_balance benchmark """

# MPI imports
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import sys
import os
from os.path import join, exists
import traceback

from optparse import OptionParser

# logging imports
import logging

# local imports
from pysph.base.kernels import CubicSplineKernel
from pysph.base.point import Point
from pysph.parallel.parallel_cell import ParallelCellManager

from pysph.base.particle_array import ParticleArray
from pysph.parallel.load_balancer import get_load_balancer_class

from pysph.solver.particle_generator import DensityComputationMode as Dcm
from pysph.solver.particle_generator import MassComputationMode as Mcm
from pysph.solver.basic_generators import RectangleGenerator, LineGenerator

LoadBalancer = get_load_balancer_class()

def parse_options(args=None):
    """parse commandline options from given list (default=sys.argv[1:])"""
    # default values
    square_width = 1.0
    np_d = 50
    particle_spacing = square_width / np_d
    particle_radius = square_width / np_d
    sph_interpolations = 1
    num_iterations = 10
    num_load_balance_iterations = 500
    max_cell_scale = 2.0

    op = OptionParser()
    op.add_option('-t', '--type', dest='type', default="square",
                  help='type of problem to load_balance, one of "dam_break" or "square"')
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
                  action="store_true", default=True, dest='verbose',
                  help='print large amounts of debug information')
    op.add_option('-c', '--max-cell-scale', dest='max_cell_scale',
                  metavar='MAX_CELL_SCALE',
                  help='specify the ratio of largest cell to smallest cell')
    
    # parse the input arguments
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
    log_filename = os.path.join(options.destdir, 'load_balance.log.%d'%rank)
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

def create_particles(options):
    if options.type == "square":
        # create the square block of particles.
        start_point = Point(0, 0, 0)
        end_point = Point(options.square_width, options.square_width, 0)
        
        parray = ParticleArray()
        if rank == 0:
            rg = RectangleGenerator(start_point=start_point,
                                    end_point=end_point,
                                    particle_spacing_x1=options.particle_spacing,
                                    particle_spacing_x2=options.particle_spacing,
                                    density_computation_mode=Dcm.Set_Constant,
                                    particle_density=1000.0,
                                    mass_computation_mode=Mcm.Compute_From_Density,
                                    particle_h=options.particle_radius,
                                    kernel=CubicSplineKernel(2),
                                    filled=True)
            tmp = rg.get_particles()
            parray.append_parray(tmp)
        
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
        
        return [parray]
    
    elif options.type == "dam_break":
        
        dam_wall = ParticleArray()
        dam_fluid = ParticleArray()
    
        if rank == 0:
                
            radius = 0.2
            dam_width=10.0
            dam_height=7.0
            solid_particle_h=radius
            dam_particle_spacing=radius/9.
            solid_particle_mass=1.0
            origin_x=origin_y=0.0
        
            fluid_particle_h=radius
            fluid_density=1000.
            fluid_column_height=3.0
            fluid_column_width=2.0
            fluid_particle_spacing=radius
    
            # generate the left wall - a line
            lg = LineGenerator(particle_mass=solid_particle_mass,
                           mass_computation_mode=Mcm.Set_Constant,
                           density_computation_mode=Dcm.Ignore,
                           particle_h=solid_particle_h,
                           start_point=Point(0, 0, 0),
                           end_point=Point(0, dam_height, 0),
                           particle_spacing=dam_particle_spacing)
            tmp = lg.get_particles()
            dam_wall.append_parray(tmp)
            
            # generate one half of the base
            lg.start_point = Point(dam_particle_spacing, 0, 0)
            lg.end_point = Point(dam_width/2, 0, 0)
            tmp = lg.get_particles()
            dam_wall.append_parray(tmp)
    
            # generate particles for the left column of fluid.
            rg = RectangleGenerator(
                start_point=Point(origin_x+2.0*solid_particle_h,
                                  origin_y+2.0*solid_particle_h,
                                  0.0),
                end_point=Point(origin_x+2.0*solid_particle_h+fluid_column_width,
                                origin_y+2.0*solid_particle_h+fluid_column_height, 0.0),
                particle_spacing_x1=fluid_particle_spacing,
                particle_spacing_x2=fluid_particle_spacing,
                density_computation_mode=Dcm.Set_Constant,
                mass_computation_mode=Mcm.Compute_From_Density,
                particle_density=1000.,
                particle_h=fluid_particle_h,
                kernel=CubicSplineKernel(2),                            
                filled=True)
            dam_fluid = rg.get_particles()
    
            # generate the right wall - a line
            lg = LineGenerator(particle_mass=solid_particle_mass,
                           mass_computation_mode=Mcm.Set_Constant,
                           density_computation_mode=Dcm.Ignore,
                           particle_h=solid_particle_h,
                           start_point=Point(dam_width, 0, 0),
                           end_point=Point(dam_width, dam_height, 0),
                           particle_spacing=dam_particle_spacing)
            
            tmp = lg.get_particles()
            dam_wall.append_parray(tmp)
            
            # generate the right half of the base
            lg.start_point = Point(dam_width/2.+dam_particle_spacing, 0, 0)
            lg.end_point = Point(dam_width, 0, 0)
            tmp = lg.get_particles()
            dam_wall.append_parray(tmp)

        for parray in [dam_fluid, dam_wall]:
        
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
        
        return [dam_fluid, dam_wall]

def create_cell_manager(options):
    print 'creating cell manager', options
    # create a parallel cell manager.
    cell_manager = ParallelCellManager(arrays_to_bin=[],
                                       max_cell_scale=options.max_cell_scale,
                                       dimension=2,
                                       load_balancing=False,
                                       initialize=False)
    # enable load balancing
    cell_manager.load_balancer = LoadBalancer(parallel_cell_manager=cell_manager)
    cell_manager.load_balancer.skip_iteration = 1
    cell_manager.load_balancer.threshold_ratio = 10.
    
    for i,pa in enumerate(create_particles(options)):
        cell_manager.arrays_to_bin.append(pa)
        print 'parray %d:'%i, pa.get_number_of_particles()

    cell_manager.initialize()
    print 'num_particles', cell_manager.get_number_of_particles()
    
    return cell_manager

def get_lb_args():
    return [
            dict(method='normal'),
            dict(method='normal', adaptive=True),
            dict(method='serial'),
            dict(method='serial', adaptive=True),
            dict(method='serial', distr_func='auto'),
            dict(method='serial', distr_func='geometric'),
            dict(method='serial_mkmeans', max_iter=200, c=0.3, t=0.2, tr=0.8, u=0.4, e=3, er=6, r=2.0),
            dict(method='serial_sfc', sfc_func_name='morton'),
            dict(method='serial_sfc', sfc_func_name='hilbert'),
            dict(method='serial_metis'),
           ]

def get_desc_name(lbargs):
    method = lbargs.get('method','')
    adaptive = lbargs.get('adaptive', False)
    if adaptive:
        method += '_a'
    sfcfunc = lbargs.get('sfc_func_name')
    if sfcfunc:
        method += '_' + sfcfunc
    redistr_method = lbargs.get('distr_func')
    if redistr_method:
        method += '_' + redistr_method
    return method
