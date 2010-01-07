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

square_width = 1.0
particle_spacing = 0.01
particle_radius = 0.01
total_iterations = 10

# logging imports
import logging
logger = logging.getLogger()
log_filename = 'log_pysph_' + str(rank)
logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w')
#logger.addHandler(logging.StreamHandler())

# local imports
from pysph.base.kernel2d import CubicSpline2D
from pysph.base.point import Point
from pysph.base.nnps import *
from pysph.parallel.parallel_cell import *
from pysph.parallel.parallel_controller import ParallelController
from pysph.parallel.parallel_component import ParallelComponent
from pysph.solver.fluid import Fluid
from pysph.solver.solver_base import *
from pysph.parallel.dummy_solver import *
from pysph.solver.particle_generator import DensityComputationMode as Dcm
from pysph.solver.particle_generator import MassComputationMode as Mcm
from pysph.solver.basic_generators import RectangleGenerator
from pysph.solver.parallel_vtk_writer import ParallelVTKWriter
from pysph.sph.sph_calc import SPHBase
from pysph.sph.density_funcs import SPHRho3D
from pysph.solver.vtk_writer import VTKWriter

# create a parallel controller.
controller = ParallelController()

# create a component manager.
component_manager = ComponentManager()
parallel_component = ParallelComponent(name='parallel_component', solver=None)
component_manager.add_component(parallel_component)

# create a parallel cell manager.
cell_manager = ParallelCellManager(parallel_controller=controller,
                                   initialize=False,
                                   solver=None)
# enable load balancing
cell_manager.load_balancing = True
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
kernel = CubicSpline2D()

# create the square block of particles.
start_point = Point(0, 0, 0)
end_point = Point(square_width, square_width, 0)

block = Fluid(name='block')
solver.add_entity(block)
parray = block.get_particle_array()
if rank == 0:
    rg = RectangleGenerator(start_point=start_point,
                            end_point=end_point,
                            particle_spacing_x1=particle_spacing,
                            particle_spacing_x2=particle_spacing,
                            density_computation_mode=Dcm.Set_Constant,
                            particle_density=1000.0,
                            mass_computation_mode=Mcm.Compute_From_Density,
                            particle_h=particle_radius,
                            kernel=kernel,
                            filled=True)
    block.add_particles(rg.get_particles())

# create a parallel vtk writer
data_file_name = 'block_data_' + str(rank)
vtk_writer = ParallelVTKWriter(solver=solver, 
                               entity_list=[block],
                               file_name_prefix=data_file_name,
                               scalars=['rho', 'pid', '_tmp', 'h', 'm', 'tag'],
                               vectors={})
component_manager.add_component(vtk_writer)

solver._setup_entities()

if rank != 0:
    # add some necessary properties to the particle array.
    parray.add_property({'name':'x'})
    parray.add_property({'name':'y'})
    parray.add_property({'name':'z'})
    parray.add_property({'name':'h', 'default':particle_radius})
    parray.add_property({'name':'rho', 'default':1000.})
    parray.add_property({'name':'pid'})
    parray.add_property({'name':'_tmp', 'default':0.0})
    parray.add_property({'name':'m'})
else:
    parray.add_property({'name':'_tmp'})
    parray.add_property({'name':'pid', 'default':0.0})

solver._setup_nnps()

# create the SPHRho3D class
parray = block.get_particle_array()
sph_func = SPHRho3D(source=parray, dest=parray)
sph_sum = SPHBase(sources=[parray], dest=parray, kernel=kernel,
                  sph_funcs=[sph_func],
                  nnps_manager=nnps_manager)
vtk_writer.setup_component()

processing_times = []
communication_times = []
total_iteration_times = []
particle_counts = []

import time

particle_counts.append(parray.get_number_of_particles())

for i in range(total_iterations):

    t1 = time.time()

    # parallel operations.
    #vtk_writer.write()
    nnps_manager.py_update()
    t2 = time.time()
    
    communication_times.append(t2-t1)
    parr = block.get_particle_array()
    t2 = time.time()

    # computation
    particle_counts.append(parr.num_real_particles)
    sph_sum.sph1('_tmp')
    parr.rho[:] = parr._tmp
    parray.set_dirty(True)
    t3 = time.time()
    processing_times.append(t3-t2)

    solver.current_iteration += 1
    t4 = time.time()

    total_iteration_times.append(t4-t1)

# write the three times into a file.
fname = 'stats_' + str(rank)

file = open(fname, 'w')

for i in range(len(total_iteration_times)):
    ln = str(total_iteration_times[i]) + ' '
    ln += str(communication_times[i]) + ' '
    ln += str(processing_times[i]) + '\n'
    file.write(ln)

file.close()

# write the particle counts
fname = 'pcount_' + str(rank)

file = open(fname, 'w')

for i in range(len(total_iteration_times)):
    file.write(str(particle_counts[i]))
    file.write('\n')

file.close()
