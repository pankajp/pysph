"""
Script for setting up a simulation where a sqaure block of fluid falls into
a tank containing fluid.

"""

###############################################################################
# setup logging
import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, filename='log_pysph',
                    format='%(module)s : %(levelname)s : %(message)s',
                    filemode='w')
logger.addHandler(logging.StreamHandler())
###############################################################################


###############################################################################
# local imports
from pysph.base.kernel2d import CubicSpline2D
from pysph.base.point import Point
from pysph.solver.fsf_solver import FSFSolver, DensityComputationMode
from pysph.solver.solid import Solid
from pysph.solver.fluid import Fluid
from pysph.solver.particle_generator import DensityComputationMode as Dcm
from pysph.solver.particle_generator import MassComputationMode as Mcm
from pysph.solver.basic_generators import RectangleGenerator
from pysph.solver.iteration_skip_component import IterationSkipComponent as\
    ISkipper
from pysph.solver.vtk_writer import VTKWriter
from pysph.solver.time_step_components import *
from pysph.solver.runge_kutta_integrator import *
###############################################################################


###############################################################################
# Parameters for the simulation.
dam_width=3.2196
dam_height=1.8
solid_particle_h=0.02
dam_particle_spacing=0.001
solid_particle_mass=1.0
origin_x=origin_y=0.0

fluid_particle_h=0.02
fluid_density=1000.
fluid_column_height=0.6
fluid_column_width=3.2196-solid_particle_h*2.0
fluid_particle_spacing=0.02

drop_size_x = 0.3
drop_size_y = 0.3

dam_center_x = (dam_width+origin_x)*0.5
dam_center_y = (dam_height+origin_y)*0.5 + 0.4
###############################################################################


# Create a solver instance using default parameters and some small changes.
solver = FSFSolver(time_step=0.00001, total_simulation_time=10.0,
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

# create fluid particles in the tank.
dam_fluid = Fluid(name='dam_fluid')
dam_fluid.properties['rho'] = 1000.
rg = RectangleGenerator(start_point=Point(origin_x+2.0*solid_particle_h,
                                          origin_y+2.0*solid_particle_h, 0.0),
                        end_point=Point(fluid_column_width,
                                        origin_y+2.0*solid_particle_h+fluid_column_height,
                                        0.0),
                        particle_spacing_x1=fluid_particle_spacing,
                        particle_spacing_x2=fluid_particle_spacing,
                        particle_density=fluid_density,
                        density_computation_mode=Dcm.Set_Constant,
                        mass_computation_mode=Mcm.Compute_From_Density,
                        particle_h=fluid_particle_h,
                        kernel=solver.kernel,
                        filled=True)
p1 = rg.get_particles()
dam_fluid.add_particles(p1, group_id=-1)

# create fluid particles for the drop
rg = RectangleGenerator(start_point=Point(dam_center_x-drop_size_x*0.5,
                                          dam_center_y-drop_size_y*0.5, 0.0),
                        end_point=Point(dam_center_x+drop_size_x*0.5,
                                        dam_center_y+drop_size_y*0.5, 0.0),
                        mass_computation_mode=Mcm.Set_Constant,
                        particle_mass=p1.m[0],
                        density_computation_mode=Dcm.Set_Constant,
                        particle_density=fluid_density,
                        particle_h=fluid_particle_h,
                        particle_spacing_x1=fluid_particle_spacing,
                        particle_spacing_x2=fluid_particle_spacing,
                        kernel=solver.kernel,
                        filled=True)

# add the particles of the drop with a different group id.
dam_fluid.add_particles(rg.get_particles(), group_id=1)
solver.add_entity(dam_fluid)

# create a vtk writer to write an output file.
vtk_writer = VTKWriter(solver=solver, entity_list=[dam_wall, dam_fluid],
                       file_name_prefix='square_drop',
                       scalars=['rho', 'p', 'm', 'rho_rate', 'group', 'h'],
                       vectors={'velocity':['u', 'v', 'w'],
                                'acceleration':['ax', 'ay', 'az'],
                                'pressure_acclr':['pacclr_x', 'pacclr_y', 'pacclr_z'],
                                'visc_acclr':['avisc_x', 'avisc_y', 'avisc_z'],
                                'boundary':['bacclr_x', 'bacclr_y', 'bacclr_z']})
it_skipper = ISkipper(solver=solver)
it_skipper.add_component(vtk_writer, skip_iteration=50)
pic = solver.component_categories['post_integration']
pic.append(it_skipper)


# add a time step component to the solver post-integration components.
ts = MonaghanKosForceBasedTimeStepComponent(solver=solver, min_time_step=0.00001, 
				  max_time_step=-1.0)
solver.component_manager.add_component(ts, notify=True)
# add it to the iteration skipper component with interval set to 5 iterations.
it_skipper.add_component(ts, skip_iteration=1)

# start the solver.
solver.solve()
