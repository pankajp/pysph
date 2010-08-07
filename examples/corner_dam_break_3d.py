"""
Script for setting up a 3D corner dam break problem.

This problem consist of modeling a 3D block of water blocked on three sides.

"""
###############################################################################
# parse command line arguements.
from optparse import OptionParser
op = OptionParser()
op.add_option('-r', '--radius', dest='radius',
	      help='interaction radius of particles')

from sys import argv
args = op.parse_args()
options = args[0]
###############################################################################


###############################################################################
# setup logging
import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, filename='log_pysph',
                    filemode='w')
logger.addHandler(logging.StreamHandler())
###############################################################################


###############################################################################
# local imports
from pysph.base.kernels import CubicSplineKernel
from pysph.base.point import Point
from pysph.solver.fsf_solver import FSFSolver
from pysph.solver.solid import Solid
from pysph.solver.fluid import Fluid
from pysph.solver.particle_generator import DensityComputationMode as Dcm
from pysph.solver.particle_generator import MassComputationMode as Mcm
from pysph.solver.basic_generators import CuboidGenerator
from pysph.solver.iteration_skip_component import IterationSkipComponent as\
    ISkipper
from pysph.solver.vtk_writer import VTKWriter
from pysph.solver.runge_kutta_integrator import *
from pysph.solver.time_step_components import MonaghanKosForceBasedTimeStepComponent
###############################################################################

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
solver = FSFSolver(time_step=0.00001, total_simulation_time=10.0,
                   kernel=CubicSplineKernel())
solver.enable_timing = True
solver.timer.output_file_name = 'times'
solver.timer.write_after=10

# UNCOMMENT THE FOLLOWING TWO LINES TO USE A DIFFERENT INTEGRATOR.
#integrator = RK2Integrator(name='integrator_default', solver=solver)
#solver.integrator = integrator

# Generate the dam wall entity
dam_wall = Solid(name='dam_wall')
cg = CuboidGenerator(start_point=Point(0, 0, 0),
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

dam_wall.add_particles(cg.get_particles())
solver.add_entity(dam_wall)

# create fluid particles
dam_fluid = Fluid(name='dam_fluid')
dam_fluid.properties['rho'] = 1000.
bkr = 2.0
start_point = Point(origin_x+bkr*solid_particle_h,
                       origin_y+bkr*solid_particle_h,
                       origin_z+bkr*solid_particle_h)

end_point = Point(origin_x+bkr*solid_particle_h+fluid_column_length,
                     origin_y+bkr*solid_particle_h+fluid_column_height,
                     origin_z+bkr*solid_particle_h+fluid_column_width)

cg = CuboidGenerator(start_point=start_point, 
		     end_point=end_point,
		     particle_spacing_x=fluid_particle_spacing,
		     particle_spacing_y=fluid_particle_spacing,
                     particle_spacing_z=fluid_particle_spacing,
	             density_computation_mode=Dcm.Set_Constant,
	  	     mass_computation_mode=Mcm.Compute_From_Density,
		     particle_h=fluid_particle_h,
		     filled=True,
		     kernel=solver.kernel,
		     particle_density=fluid_density)

dam_fluid.add_particles(cg.get_particles())
solver.add_entity(dam_fluid)

# create a vtk writer to write an output file.
vtk_writer = VTKWriter(solver=solver, entity_list=[dam_wall, dam_fluid],
                       file_name_prefix='corner_db',
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

# add a timestep modifying component.
ts = MonaghanKosForceBasedTimeStepComponent(solver=solver, min_time_step=0.00001,
                                         max_time_step=-1.0)
solver.component_manager.add_component(ts, notify=True)
it_skipper.add_component(ts, skip_iteration=1)

# start the solver.
solver.solve()
