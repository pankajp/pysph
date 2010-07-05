"""
Script for setting up a 2d dam break problem.
"""
# setup logging
import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG, filename='/tmp/log_pysph',
                    filemode='w')
logger.addHandler(logging.StreamHandler())

# local imports
from pysph.base.kernel2d import CubicSpline2D
from pysph.base.point import Point
from pysph.solver.fsf_solver import FSFSolver
from pysph.solver.solid import Solid
from pysph.solver.fluid import Fluid
from pysph.solver.particle_generator import DensityComputationMode as Dcm
from pysph.solver.particle_generator import MassComputationMode as Mcm
from pysph.solver.basic_generators import RectangleGenerator
from pysph.solver.iteration_skip_component import IterationSkipComponent as\
    ISkipper
from pysph.solver.vtk_writer import VTKWriter
from pysph.solver.xsph_integrator import *
from pysph.solver.runge_kutta_integrator import RK2Integrator

# Parameters for the simulation.
dam_width=3.2196
dam_height=1.
solid_particle_h=0.1
dam_particle_spacing=0.1
solid_particle_mass=1.0
origin_x=origin_y=0.0

fluid_particle_h=0.1
fluid_density=1000.
fluid_column_height=0.5
fluid_column_width=2.0
fluid_particle_spacing=0.05

# Create a solver instance using default parameters and some small changes.
solver = FSFSolver(time_step=0.0001, total_simulation_time=10.0,
                   kernel=CubicSpline2D())
integrator = RK2Integrator(name='integrator_default', solver=solver)
solver.integrator = integrator


# Generate the dam wall entity
dam_wall = Solid(name='dam_wall')
rg = RectangleGenerator(start_point=Point(0, 0, 0),
                        end_point=Point(dam_width, dam_height, 0.0),
                        particle_spacing_x1=dam_particle_spacing,
                        particle_spacing_x2=dam_particle_spacing,
                        density_computation_mode=Dcm.Ignore,
                        mass_computation_mode=Mcm.Set_Constant,
                        particle_h=solid_particle_h,
                        particle_mass=solid_particle_mass,
                        filled=False)
dam_wall.add_particles(rg.get_particles())
solver.add_entity(dam_wall)

# create fluid particles
dam_fluid = Fluid(name='dam_fluid')
dam_fluid.properties.rho = 1000.
rg2 = RectangleGenerator(start_point=Point(origin_x+2.0*solid_particle_h,
                            origin_y+2.0*solid_particle_h, 0.0),
                        end_point=Point(origin_x+2.0*solid_particle_h+fluid_column_width,
                            origin_y+2.0*solid_particle_h+fluid_column_height, 0.0),
                        density_computation_mode = Dcm.Set_Constant,
                        particle_density=fluid_density,
                        mass_computation_mode=Mcm.Compute_From_Density,
                        particle_spacing_x1=fluid_particle_spacing,
                        particle_spacing_x2=fluid_particle_spacing,
                        particle_h=fluid_particle_h,
                        kernel=solver.kernel,
                        filled=True)
dam_fluid.add_particles(rg2.get_particles())
solver.add_entity(dam_fluid)

# create a vtk writer to write an output file.
vtk_writer = VTKWriter(solver=solver, entity_list=[dam_wall, dam_fluid],
                       file_name_prefix='dam_break_2d/dam_break_2d',
                       scalars=['rho', 'p', 'm', 'rho_rate'],
                       vectors={'velocity':['u', 'v', 'w'],
                                'acceleration':['ax', 'ay', 'az'],
                                'pressure_acclr':['pacclr_x', 'pacclr_y', 'pacclr_z'],
                                'visc_acclr':['avisc_x', 'avisc_y', 'avisc_z'],
                                'boundary':['bacclr_x', 'bacclr_y', 'bacclr_z']})
it_skipper = ISkipper(solver=solver)
it_skipper.add_component(vtk_writer, skip_iteration=20)
pic = solver.component_categories['post_integration']
pic.append(it_skipper)

# start the solver.
solver.solve()
