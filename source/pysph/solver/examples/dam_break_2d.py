"""
Script for setting up a 2d dam break problem.
"""

# FIXME: 
#
# 1. particle_group_id ought to be set by the particle generator and not managed
# by the solver in the add_particles call.
#
# 2. create_entity should return an entity which can have an add_particles
# method.



# local imports
from pysph.solver.fsf_solver import FSFSolver

# Parameters for the simulation.
dam_width=3.2196
dam_height=1.
solid_particle_h=0.1
dam_particle_spacing=0.1
origin_x=origin_y=0.0

fluid_particle_h=0.1
fluid_density=1000.
fluid_column_height=0.5
fluid_column_width=2.0
fluid_particle_spacing=0.05

# Create a solver instance using default parameters and some small changes.
solver = FSFSolver(time_step=0.0001, total_simulation_time=10.0)

# Generate the dam wall entity
solver.create_entity(entity_type='solid', entity_name='dam_wall')

# Create the particles for the wall entity, note that wall_particles is a
# ParticleArray
wall_particles = generate_2d_box(
    left_x=origin_x, left_y=origin_y,
    right_x=origin_x+dam_width, right_y=origin_y+dam_height,
    particle_spacing_x=dam_particle_spacing,
    particle_spacing_y=dam_particle_spacing,
    end_points_exact=True)

# Add the particles to the dam_wall entity.
solver.add_particles(entity='dam_wall', wall_particles,
                     particle_group_id=0)

# generate the fluid particles
fluid_bottom_x = origin_x + 2.0*solid_particle_h
fluid_bottom_y = origin_y + 2.0*solid_particle_h
fluid_top_x = fluid_bottom_x + fluid_column_width
fluid_top_y = fluid_bottom_y + fluid_column_height

solver.create_entity(entity_type='fluid', entity_name='dam_fluid',
                  rest_density=1000.0, variable_h=False, h=fluid_particle_h)
fluid_particles = particle_generators.generate_rectangle(
    filled=True,
    left_x=fluid_bottom_x, left_y=fluid_bottom_y,
    right_x=fluid_top_x, right_y=fluid_top_y,
    particle_spacing_x=fluid_particle_spacing,
    particle_spacing_y=fluid_particle_spacing,
    end_points_exact=True)
solver.add_particles(entity='dam_fluid', fluid_particles,
                     particle_group_id=0)

# Now simulate the problem.
solver.solve()
