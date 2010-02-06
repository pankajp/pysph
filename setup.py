"""A general purpose Smoothed Particle Hydrodynamics framework.

The SPH package provides a general purpose framework for SPH simulations
in Python.  The framework emphasizes flexibility and efficiency while
allowing most of the user code to be written in pure Python.
"""

from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
import numpy

inc_dir = numpy.get_include()

mpi_inc_dir = ['/usr/include/mpi', '/usr/local/include/mpi', 
               '/opt/local/include/mpi']

# base extension modules.
base = [Extension("pysph.base.attrdict",
                  ["source/pysph/base/attrdict.pyx"], include_dirs=[inc_dir]),
        Extension("pysph.base.carray", 
                  ["source/pysph/base/carray.pyx"], include_dirs=[inc_dir]),
        Extension("pysph.base.point",
                  ["source/pysph/base/point.pyx"], include_dirs=[inc_dir]),
        Extension("pysph.base.plane",
                  ["source/pysph/base/plane.pyx"], include_dirs=[inc_dir]),
        Extension("pysph.base.particle_array",
                  ["source/pysph/base/particle_array.pyx"], include_dirs=[inc_dir]),
        Extension("pysph.base.cell",
                  ["source/pysph/base/cell.pyx"], include_dirs=[inc_dir]),
        Extension("pysph.base.polygon_array",
                  ["source/pysph/base/polygon_array.pyx"], include_dirs=[inc_dir]),
        Extension("pysph.base.nnps",
                  ["source/pysph/base/nnps.pyx"], include_dirs=[inc_dir]),
        Extension("pysph.base.particle_tags",
                  ["source/pysph/base/particle_tags.pyx"], include_dirs=[inc_dir]),
	] 

sph = [
        Extension("pysph.sph.sph_func",
                  ["source/pysph/sph/sph_func.pyx"], include_dirs=[inc_dir]),
        Extension("pysph.sph.sph_calc",
                  ["source/pysph/sph/sph_calc.pyx"], include_dirs=[inc_dir]),
        Extension("pysph.sph.misc_particle_funcs",
                  ["source/pysph/sph/misc_particle_funcs.pyx"], include_dirs=[inc_dir]),
        Extension("pysph.sph.density_funcs",
                  ["source/pysph/sph/density_funcs.pyx"], include_dirs=[inc_dir]),
        Extension("pysph.sph.basic_funcs",
                  ["source/pysph/sph/basic_funcs.pyx"], include_dirs=[inc_dir]),
        Extension("pysph.sph.pressure_funcs",
                  ["source/pysph/sph/pressure_funcs.pyx"], include_dirs=[inc_dir])
        ]

parallel = [
    Extension("pysph.parallel.parallel_controller",
              ["source/pysph/parallel/parallel_controller.pyx"],
              include_dirs=[inc_dir] + mpi_inc_dir),
    Extension("pysph.parallel.cy_parallel_cell",
              ["source/pysph/parallel/cy_parallel_cell.pyx"],
              include_dirs=[inc_dir] + mpi_inc_dir),
    ]

# kernel extension modules.
kernels = [Extension("pysph.base.kernelbase",
                     ["source/pysph/base/kernelbase.pyx"], include_dirs=[inc_dir]),
           Extension("pysph.base.kernel1d",
                     ["source/pysph/base/kernel1d.pyx"], include_dirs=[inc_dir]),
           Extension("pysph.base.kernel2d",
                     ["source/pysph/base/kernel2d.pyx"], include_dirs=[inc_dir]),
           Extension("pysph.base.kernel3d",
                     ["source/pysph/base/kernel3d.pyx"], include_dirs=[inc_dir]),
           ]

solver = [Extension("pysph.solver.fast_utils",
                    ["source/pysph/solver/fast_utils.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.typed_dict",
                    ["source/pysph/solver/typed_dict.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.time_step",
                    ["source/pysph/solver/time_step.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.base",
                    ["source/pysph/solver/base.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.geometry",
                    ["source/pysph/solver/geometry.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.entity_types",
                    ["source/pysph/solver/entity_types.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.entity_base",
                    ["source/pysph/solver/entity_base.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.fluid",
                    ["source/pysph/solver/fluid.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.solid",
                    ["source/pysph/solver/solid.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.solver_base",
                    ["source/pysph/solver/solver_base.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.integrator_base",
                    ["source/pysph/solver/integrator_base.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.runge_kutta_integrator",
                    ["source/pysph/solver/runge_kutta_integrator.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.speed_of_sound",
                    ["source/pysph/solver/speed_of_sound.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.sph_component",
                    ["source/pysph/solver/sph_component.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.pressure_components",
                    ["source/pysph/solver/pressure_components.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.pressure_gradient_components",
                    ["source/pysph/solver/pressure_gradient_components.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.xsph_component",
                    ["source/pysph/solver/xsph_component.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.xsph_integrator",
                    ["source/pysph/solver/xsph_integrator.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.particle_generator",
                    ["source/pysph/solver/particle_generator.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.iteration_skip_component",
                    ["source/pysph/solver/iteration_skip_component.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.file_writer_component",
                    ["source/pysph/solver/file_writer_component.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.viscosity_components",
                    ["source/pysph/solver/viscosity_components.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.boundary_force_components",
                    ["source/pysph/solver/boundary_force_components.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.nnps_updater",
                    ["source/pysph/solver/nnps_updater.pyx"], include_dirs=[inc_dir]),
          Extension("pysph.solver.time_step_components",
                    ["source/pysph/solver/time_step_components.pyx"], include_dirs=[inc_dir])
          ]

# all extension modules.
ext_modules = base +\
    kernels +\
    sph+\
    parallel + \
    solver

setup(name='pysph',
      version = '0.9',
      author = 'Chandrashekhar P. Kaushik',
      author_email = 'shekhar.kaushik@iitb.ac.in',
      description = "A general purpose Smoothed Particle Hydrodynamics framework",
      long_description = __doc__,
      license = "BSD",
      keywords = "SPH simulation computational fluid dynamics",
      test_suite = "nose.collector",
      packages = find_packages('source'),
      package_dir = {'pysph': 'source/pysph'},

      ext_modules = ext_modules,
      
      include_package_data = True,
      cmdclass={'build_ext': build_ext},
      )
