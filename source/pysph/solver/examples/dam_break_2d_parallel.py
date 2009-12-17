#!/usr/bin/env python
"""
Script for setting up a 2d dam break problem.
"""

# MPI imports
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# logging imports
import logging
logger = logging.getLogger()
filename = 'log_pysph_'+str(rank)
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
from pysph.solver.vtk_writer import VTKWriter

#pcm = ParallelCellManager(initialize=False)


# parameters for the simulation
dam_width=3.2196
dam_height=3.
solid_particle_h=0.05
dam_particle_spacing=0.001
solid_particle_mass=1.0
origin_x=origin_y=0.0

fluid_particle_h=0.05
fluid_density=1000.
fluid_column_height=1.5
fluid_column_width=1.0
fluid_particle_spacing=0.05

# Create a solver instance using default parameters and some small changes.
solver = ParallelFsfSolver(cell_manager=None, time_step=0.0001,
                           total_simulation_time=10.0, kernel=CubicSpline2D())
solver.timer.output_file_name = 'times_'+str(rank)
solver.enable_timing=True
solver.timer.write_after = 50

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

elif rank == 1:
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

    # generate particles for the right column of fluid.
    dam_fluid=Fluid(name='dam_fluid')
    rg = RectangleGenerator(
        start_point=Point(dam_width-2.0*solid_particle_h-fluid_column_width,
                          origin_y+2.0*solid_particle_h,
                          0.0),
        end_point=Point(dam_width-2.0*solid_particle_h,
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

# add entity to solver - may be empty.
solver.add_entity(dam_wall)
solver.add_entity(dam_fluid)

file_name_prefix='tc_'+str(rank)
vtk_writer = VTKWriter(solver=solver, entity_list=[dam_wall, dam_fluid],
                       file_name_prefix=file_name_prefix,
                       scalars=['rho', 'p', 'm', 'rho_rate', 'tag', 'local'],
                       vectors={'velocity':['u', 'v', 'w']})

#parr_vtk_writer = ParallelVTKWriter(rank=rank, collector=0)

# add the parr_vtk_writer to an iteration skipper component.
iter_skipper = IterSkipper(solver=solver)
iter_skipper.add_component(vtk_writer, skip_iteration=50)
pic = solver.component_categories['pre_integration']
pic.append(iter_skipper)

# start the solver.
solver.solve()
