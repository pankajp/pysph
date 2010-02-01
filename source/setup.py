"""A general purpose Smoothed Particle Hydrodynamics framework.

The SPH package provides a general purpose framework for SPH simulations
in Python.  The framework emphasizes flexibility and efficiency while
allowing most of the user code to be written in pure Python.
"""

from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext

# base extension modules.
base = [Extension("pysph.base.attrdict",
                  ["pysph/base/attrdict.pyx"]),
        Extension("pysph.base.carray", 
                  ["pysph/base/carray.pyx"]),
        Extension("pysph.base.point",
                  ["pysph/base/point.pyx"]),
        Extension("pysph.base.plane",
                  ["pysph/base/plane.pyx"]),
        Extension("pysph.base.particle_array",
                  ["pysph/base/particle_array.pyx"]),
        Extension("pysph.base.cell",
                  ["pysph/base/cell.pyx"]),
        Extension("pysph.base.polygon_array",
                  ["pysph/base/polygon_array.pyx"]),
        Extension("pysph.base.nnps",
                  ["pysph/base/nnps.pyx"]),
        Extension("pysph.base.particle_tags",
                  ["pysph/base/particle_tags.pyx"]),
	] 

sph = [
        Extension("pysph.sph.sph_func",
                  ["pysph/sph/sph_func.pyx"]),
        Extension("pysph.sph.sph_calc",
                  ["pysph/sph/sph_calc.pyx"]),
        Extension("pysph.sph.misc_particle_funcs",
                  ["pysph/sph/misc_particle_funcs.pyx"]),
        Extension("pysph.sph.density_funcs",
                  ["pysph/sph/density_funcs.pyx"]),
        Extension("pysph.sph.basic_funcs",
                  ["pysph/sph/basic_funcs.pyx"]),
        Extension("pysph.sph.pressure_funcs",
                  ["pysph/sph/pressure_funcs.pyx"])
        ]

parallel = [
    Extension("pysph.parallel.parallel_controller",
              ["pysph/parallel/parallel_controller.pyx"],
              include_dirs=['/usr/include/mpi']),
    Extension("pysph.parallel.cy_parallel_cell",
              ["pysph/parallel/cy_parallel_cell.pyx"],
              include_dirs=['/usr/include/mpi'])
    ]

# kernel extension modules.
kernels = [Extension("pysph.base.kernelbase",
                     ["pysph/base/kernelbase.pyx"]),
           Extension("pysph.base.kernel1d",
                     ["pysph/base/kernel1d.pyx"]),
           Extension("pysph.base.kernel2d",
                     ["pysph/base/kernel2d.pyx"]),
           Extension("pysph.base.kernel3d",
                     ["pysph/base/kernel3d.pyx"]),
           ]

solver = [Extension("pysph.solver.fast_utils",
                    ["pysph/solver/fast_utils.pyx"]),
          Extension("pysph.solver.typed_dict",
                    ["pysph/solver/typed_dict.pyx"]),
          Extension("pysph.solver.time_step",
                    ["pysph/solver/time_step.pyx"]),
          Extension("pysph.solver.base",
                    ["pysph/solver/base.pyx"]),
          Extension("pysph.solver.geometry",
                    ["pysph/solver/geometry.pyx"]),
          Extension("pysph.solver.entity_types",
                    ["pysph/solver/entity_types.pyx"]),
          Extension("pysph.solver.entity_base",
                    ["pysph/solver/entity_base.pyx"]),
          Extension("pysph.solver.fluid",
                    ["pysph/solver/fluid.pyx"]),
          Extension("pysph.solver.solid",
                    ["pysph/solver/solid.pyx"]),
          Extension("pysph.solver.solver_base",
                    ["pysph/solver/solver_base.pyx"]),
          Extension("pysph.solver.integrator_base",
                    ["pysph/solver/integrator_base.pyx"]),
          Extension("pysph.solver.runge_kutta_integrator",
                    ["pysph/solver/runge_kutta_integrator.pyx"]),
          Extension("pysph.solver.speed_of_sound",
                    ["pysph/solver/speed_of_sound.pyx"]),
          Extension("pysph.solver.sph_component",
                    ["pysph/solver/sph_component.pyx"]),
          Extension("pysph.solver.pressure_components",
                    ["pysph/solver/pressure_components.pyx"]),
          Extension("pysph.solver.pressure_gradient_components",
                    ["pysph/solver/pressure_gradient_components.pyx"]),
          Extension("pysph.solver.xsph_component",
                    ["pysph/solver/xsph_component.pyx"]),
          Extension("pysph.solver.xsph_integrator",
                    ["pysph/solver/xsph_integrator.pyx"]),
          Extension("pysph.solver.particle_generator",
                    ["pysph/solver/particle_generator.pyx"]),
          Extension("pysph.solver.iteration_skip_component",
                    ["pysph/solver/iteration_skip_component.pyx"]),
          Extension("pysph.solver.file_writer_component",
                    ["pysph/solver/file_writer_component.pyx"]),
          Extension("pysph.solver.viscosity_components",
                    ["pysph/solver/viscosity_components.pyx"]),
          Extension("pysph.solver.boundary_force_components",
                    ["pysph/solver/boundary_force_components.pyx"]),
          Extension("pysph.solver.nnps_updater",
                    ["pysph/solver/nnps_updater.pyx"]),
          Extension("pysph.solver.time_step_components",
                    ["pysph/solver/time_step_components.pyx"])
          ]

# all extension modules.
ext_modules = base +\
    kernels +\
    sph+\
    parallel + \
    solver

setup(name='pysph',
      version = '0.1',
      author = 'Chandrashekhar P. Kaushik',
      author_email = 'shekhar.kaushik@iitb.ac.in',
      description = "A general purpose Smoothed Particle Hydrodynamics framework",
      long_description = __doc__,
      license = "Undecided",
      keywords = "SPH simulation computational fluid dynamics",
      test_suite = "nose.collector",
      packages = find_packages(),
      
      ext_modules = ext_modules,
      
      include_package_data = True,
      cmdclass={'build_ext': build_ext},
      )
