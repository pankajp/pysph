"""
Module to get various timings of the parallel code.

What to do ?
 
    - Create a 2d square of particles.
    - Run SPHRho3D on it and find the time it takes for each iteration. 
"""

# library imports
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
from pysph.base.cell import CellManager
from pysph.solver.fluid import Fluid
from pysph.solver.solver_base import SolverBase, ComponentManager
from pysph.solver.particle_generator import DensityComputationMode as Dcm
from pysph.solver.particle_generator import MassComputationMode as Mcm
from pysph.solver.basic_generators import RectangleGenerator
from pysph.sph.sph_calc import SPHBase
from pysph.sph.density_funcs import SPHRho3D
from pysph.solver.vtk_writer import VTKWriter


class DummySolver(SolverBase):
    """A dummy solver for serial timings code"""
    def __init__(self, component_manager=None,
                 cell_manager=None,
                 nnps_manager=None,
                 kernel=None,
                 integrator=None,
                 time_step=0.0,
                 total_simulation_time=0.0):
        SolverBase.__init__(self, component_manager=component_manager,
                            cell_manager=cell_manager,
                            nnps_manager=nnps_manager,
                            kernel=kernel)
        pass
    
    def _setup_nnps(self):
        """
        """
        for e in self.entity_list:
            e.add_arrays_to_cell_manager(self.cell_manager)

        min_cell_size, max_cell_size = self._compute_cell_sizes()

        if min_cell_size != -1 and max_cell_size != -1:
            self.cell_manager.min_cell_size = 2.*min_cell_size
            self.cell_manager.max_cell_size = 4.*min_cell_size
            
        # initialize the cell manager.
        self.cell_manager.initialize()


def parse_options(args=None):
    """parse commandline options from given list (default=sys.argv[1:])"""
    # default values
    square_width = 1.0
    particle_spacing = 0.01
    particle_radius = 0.01
    sph_interpolations = 1
    num_iterations = 10
    
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
    op.add_option('-o', '--write-vtk', 
                  action="store_true", default=False, dest='write_vtk',
                  help='write a vtk file after all iterations are done')
    op.add_option('-v', '--verbose',
                  action="store_true", default=False, dest='verbose',
                  help='print large amounts of debug information')
    
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
    
    # write the settings file
    settings_file = os.path.join(options.destdir, 'serial_timings_settings.txt')
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
    """sets up and returns a solver"""
    
    # create a component manager.
    component_manager = ComponentManager()
    
    # create a cell manager.
    cell_manager = CellManager(initialize=False)
    
    # create a nnps manager
    nnps_manager = NNPSManager(cell_manager=cell_manager)
    
    # create a dummy solver - serves no purpose but for a place holder
    solver = DummySolver(cell_manager=cell_manager, nnps_manager=nnps_manager)
                        
    # the kernel to be used
    solver.kernel = kernel = CubicSplineKernel()
    
    # create the square block of particles.
    start_point = Point(0, 0, 0)
    end_point = Point(options.square_width, options.square_width, 0)
    
    block = Fluid(name='block')
    solver.add_entity(block)
    parray = block.get_particle_array()
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
    
    # create a vtk writer
    data_file_name = 'block_data_'
    vtk_writer = VTKWriter(solver=solver, 
                           entity_list=[block],
                           file_name_prefix=data_file_name,
                           scalars=['rho','_tmp','h','m','tag'],
                           vectors={})
    component_manager.add_component(vtk_writer)
    
    solver._setup_entities()
    
    parray.add_property({'name':'_tmp'})
    
    solver._setup_nnps()
    
    vtk_writer.setup_component()
    options.vtk_writer = vtk_writer
    
    return solver


def serial_timings(args):
    """returns a dictionary of serial code timings"""
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
    
    N = options.num_iterations
    for i in range(N):
    
        t1 = time()
    
        solver.nnps_manager.py_update()
        # perform an explicit cell manager update, else the update gets called
        # in the portion where the computation time is being measure.
        solver.cell_manager.update()
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
    fname = os.path.join(options.destdir, 'stats')
    file = open(fname, 'w')
    for i in range(len(total_iteration_times)):
        ln = str(total_iteration_times[i]) + ' '
        ln += str(communication_times[i]) + ' '
        ln += str(processing_times[i]) + '\n'
        file.write(ln)
    file.close()
    
    # write the particle counts
    fname = os.path.join(options.destdir, 'pcount')
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


funcs = [serial_timings]


def bench(args=None):
    """return a list of a dictionary of serial benchmark timings"""
    if args is None:
        args = []
    timings = []
    for func in funcs:
        timings.append(func(args))
    return timings
    
if __name__ == '__main__':
    timings = bench(sys.argv[1:])
    print timings