#!/usr/bin/env python
"""
Script for setting up a 3d corner dam break problem in parallel.

"""
###############################################################################
# MPI imports
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
###############################################################################

###############################################################################
# parse command line arguments
from optparse import OptionParser
op = OptionParser()
d_help='the directory where outputs/logs are stored.\n'
d_help +=  'The directory path should be relative to the HOME directory'
op.add_option('-d', '--destdir', dest='destdir',
              help=d_help,
              metavar='DESTDIR')
op.add_option('-r', '--radius', dest='radius',
              help='interaction radius of particles')

from sys import argv
args = op.parse_args()
options = args[0]
if options.destdir is None:
    print 'A destination directory is needed'
    import sys
    sys.exit(1)

# make sure the dest directory exists, if not create it
import os
from os.path import join, exists
home = os.environ['HOME']
destdir = home+'/'+options.destdir+'/'

if not exists(destdir):
    os.mkdir(destdir)
###############################################################################


###############################################################################
# logging imports
import logging
logger = logging.getLogger()
filename = destdir+'/'+'log_pysph_'+str(rank)
logging.basicConfig(level=logging.INFO, filename=filename, filemode='w')
logger.addHandler(logging.StreamHandler())
###############################################################################

###############################################################################
# local imports
from pysph.base.kernel2d import CubicSpline2D
from pysph.base.point import Point
from pysph.parallel.parallel_fsf_solver import ParallelFsfSolver
from pysph.solver.solid import Solid
from pysph.solver.fluid import Fluid
from pysph.solver.particle_generator import DensityComputationMode as Dcm
from pysph.solver.particle_generator import MassComputationMode as Mcm
from pysph.solver.basic_generators import CuboidGenerator
from pysph.solver.iteration_skip_component import IterationSkipComponent as\
    IterSkipper
from pysph.solver.parallel_vtk_writer import ParallelVTKWriter
from pysph.parallel.load_balancer_component import LoadBalancerComponent
###############################################################################

# parameters for the simulation
if options.radius is None:
    radius = 0.2
else:
    radius = float(options.radius)


###############################################################################
# Parameters for the simulation.
# x-extent
dam_length = 7.0
# y-extent
dam_depth= 5.0
# z-extent
dam_width = 7.0

solid_particle_h=radius
dam_particle_spacing=radius/2.0
solid_particle_mass=1.0
origin_x=origin_y=origin_z=0.0

# fluid parameters.
fluid_particle_h=radius
fluid_density=1000.

# x-extent
fluid_column_length=2.0
# y-extent
fluid_column_height=2.0
# z-extent
fluid_column_width=2.0

fluid_particle_spacing=radius
###############################################################################

# Create a solver instance using default parameters and some small changes.
solver = ParallelFsfSolver(cell_manager=None, time_step=0.00001,
                           total_simulation_time=10., kernel=CubicSpline2D())
#solver.integrator=RK2Integrator(name='rk2_integrator', solver=solver)

solver.timer.output_file_name = destdir+'times_'+str(rank)
solver.enable_timing=True
solver.timer.write_after = 1
solver.cell_manager.load_balancing = True
solver.cell_manager.dimension = 3
lb = solver.cell_manager.load_balancer
lb.skip_iteration = 5
lb.threshold_ratio = 5.
# create the two entities.
dam_wall = Solid(name='dam_wall')
dam_fluid=Fluid(name='dam_fluid')

if rank == 0:
    rg = CuboidGenerator(start_point=Point(0, 0, 0),
                     end_point=Point(dam_length, dam_depth, dam_width),
                     particle_spacing_x=dam_particle_spacing,
                     particle_spacing_y=dam_particle_spacing,
                     particle_spacing_z=dam_particle_spacing,
                     density_computation_mode=Dcm.Ignore,
                     mass_computation_mode=Mcm.Set_Constant,
                     particle_h=solid_particle_h,
                     particle_mass=solid_particle_mass,
                     filled=False,
                     exclude_top=False)

    dam_wall.add_particles(rg.get_particles())

    # create fluid particles
    dam_fluid = Fluid(name='dam_fluid')
    dam_fluid.properties.rho = 1000.
    bkr = 1.5
    rg = CuboidGenerator(start_point= Point(origin_x+bkr*solid_particle_h,
                                            origin_y+bkr*solid_particle_h,
                                            origin_z+bkr*solid_particle_h),
                         end_point= Point(origin_x+bkr*solid_particle_h+fluid_column_length,
                                          origin_y+bkr*solid_particle_h+fluid_column_height,
                                          origin_z+bkr*solid_particle_h+fluid_column_width),
                         particle_spacing_x=fluid_particle_spacing,
                         particle_spacing_y=fluid_particle_spacing,
                         particle_spacing_z=fluid_particle_spacing,
                         density_computation_mode=Dcm.Set_Constant,
                         particle_density=fluid_density,
                         mass_computation_mode=Mcm.Compute_From_Density,
                         particle_h=fluid_particle_h,
                         kernel=solver.kernel,
                         filled=True)
                         
    dam_fluid.add_particles(rg.get_particles())


# add entity to solver - may be empty.
solver.add_entity(dam_wall)
solver.add_entity(dam_fluid)

file_name_prefix=destdir+'tc_'+str(rank)
vtk_writer = ParallelVTKWriter(solver=solver, entity_list=[dam_wall, dam_fluid],
                               file_name_prefix=file_name_prefix,
                               scalars=['rho', 'p', 'm', 'rho_rate', 'pid'],
                               vectors={'velocity':['u', 'v', 'w']})

# add the parr_vtk_writer to an iteration skipper component.
iter_skipper = IterSkipper(solver=solver)
iter_skipper.add_component(vtk_writer, skip_iteration=10)
pic = solver.component_categories['post_integration']
pic.append(iter_skipper)

# add a pre-iteration component to do load balancing.
lb = LoadBalancerComponent(name='load_balancer_component', solver=solver, num_iterations=1000)
pic = solver.component_categories['pre_iteration']
pic.append(lb)

# start the solver.
solver.solve()
