#!/usr/bin/env python
"""
Script for setting up a 2d dam break problem.
"""

# MPI imports
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

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

# logging imports
import logging
logger = logging.getLogger()
filename = destdir+'/'+'log_pysph_'+str(rank)
logging.basicConfig(level=logging.INFO, filename=filename, filemode='w')
logger.addHandler(logging.StreamHandler())

# local imports
from pysph.base.kernel2d import CubicSpline2D
from pysph.base.point import Point
from pysph.parallel.parallel_fsf_solver import ParallelFsfSolver
from pysph.solver.solid import Solid
from pysph.solver.fluid import Fluid
from pysph.solver.particle_generator import DensityComputationMode as Dcm
from pysph.solver.particle_generator import MassComputationMode as Mcm
from pysph.solver.basic_generators import RectangleGenerator, LineGenerator
from pysph.solver.iteration_skip_component import IterationSkipComponent as\
    IterSkipper
from pysph.solver.time_step_components import MonaghanKosForceBasedTimeStepComponent
from pysph.solver.parallel_time_step_update import ParallelTimeStepUpdateComponent
from pysph.solver.parallel_vtk_writer import ParallelVTKWriter
from pysph.solver.runge_kutta_integrator import RK2Integrator

# parameters for the simulation
if options.radius is None:
    radius = 0.005
else:
    radius = options.radius

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

# Create a solver instance using default parameters and some small changes.
solver = ParallelFsfSolver(cell_manager=None, time_step=0.0001,
                           total_simulation_time=10., kernel=CubicSpline2D())
solver.integrator=RK2Integrator(name='rk2_integrator', solver=solver)

solver.timer.output_file_name = destdir+'times_'+str(rank)
solver.enable_timing=True
solver.timer.write_after = 50
solver.cell_manager.load_balancing = True
solver.cell_manager.dimension = 2
# create the two entities.
dam_wall = Solid(name='dam_wall')
dam_fluid=Fluid(name='dam_fluid')

if rank == 0:
    # generate the left wall - a line
    lg = LineGenerator(particle_mass=solid_particle_mass,
                       mass_computation_mode=Mcm.Set_Constant,
                       density_computation_mode=Dcm.Ignore,
                       particle_h=solid_particle_h,
                       start_point=Point(0, 0, 0),
                       end_point=Point(0, dam_height, 0),
                       particle_spacing=dam_particle_spacing)

    dam_wall.add_particles(lg.get_particles())
    
    # generate one half of the base
    lg.start_point = Point(dam_particle_spacing, 0, 0)
    lg.end_point = Point(dam_width/2, 0, 0)
    dam_wall.add_particles(lg.get_particles())

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
        kernel=solver.kernel,                            
        filled=True)
    dam_fluid.add_particles(rg.get_particles())

if rank == 0:
    # generate the right wall - a line
    lg = LineGenerator(particle_mass=solid_particle_mass,
                       mass_computation_mode=Mcm.Set_Constant,
                       density_computation_mode=Dcm.Ignore,
                       particle_h=solid_particle_h,
                       start_point=Point(dam_width, 0, 0),
                       end_point=Point(dam_width, dam_height, 0),
                       particle_spacing=dam_particle_spacing)

    dam_wall.add_particles(lg.get_particles())
    
    # generate the right half of the base
    lg.start_point = Point(dam_width/2.+dam_particle_spacing, 0, 0)
    lg.end_point = Point(dam_width, 0, 0)
    dam_wall.add_particles(lg.get_particles())

# add entity to solver - may be empty.
solver.add_entity(dam_wall)
solver.add_entity(dam_fluid)

file_name_prefix=destdir+'tc_'+str(rank)
vtk_writer = ParallelVTKWriter(solver=solver, entity_list=[dam_wall, dam_fluid],
                               file_name_prefix=file_name_prefix,
                               scalars=['rho', 'p', 'm', 'rho_rate', 'tag', 'local'],
                               vectors={'velocity':['u', 'v', 'w']})

# add a time step component as a pre-integration component.
ts = MonaghanKosForceBasedTimeStepComponent(name='ts_update', solver=solver, min_time_step=0.00001,
						max_time_step=-1.0)
solver.component_manager.add_component(ts, notify=True)
pts = ParallelTimeStepUpdateComponent(name='parallel_ts_update', solver=solver, time_step_component=ts)
solver.component_manager.add_component(pts)

# add the parr_vtk_writer to an iteration skipper component.
iter_skipper = IterSkipper(solver=solver)
iter_skipper.add_component(vtk_writer, skip_iteration=50)
pic = solver.component_categories['post_integration']
pic.append(iter_skipper)
pic.append(pts)

# start the solver.
solver.solve()
