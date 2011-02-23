"""A general purpose Smoothed Particle Hydrodynamics framework.

The SPH package provides a general purpose framework for SPH simulations
in Python.  The framework emphasizes flexibility and efficiency while
allowing most of the user code to be written in pure Python.
"""

from setuptools import find_packages, setup
from Cython.Distutils import build_ext
from numpy.distutils.extension import Extension

import numpy
import os

HAS_MPI4PY = True
try:
    import mpi4py
except ImportError:
    HAS_MPI4PY = False

inc_dirs = [numpy.get_include()]
if HAS_MPI4PY:
    inc_dirs += [mpi4py.get_include()]

if 'MPI_INCLUDE' in os.environ:
    mpi_inc_dir = [os.environ['MPI_INCLUDE']]
else:
    mpi_inc_dir = ['/usr/include/mpi', '/usr/local/include/mpi', 
               '/opt/local/include/mpi']

# base extension modules.
base = [Extension("pysph.base.carray", 
                  ["source/pysph/base/carray.pyx"], include_dirs=inc_dirs),

        Extension("pysph.base.point",
                  ["source/pysph/base/point.pyx"], include_dirs=inc_dirs),

        Extension("pysph.base.plane",
                  ["source/pysph/base/plane.pyx"], include_dirs=inc_dirs),

        Extension("pysph.base.particle_array",
                  ["source/pysph/base/particle_array.pyx"], 
                  include_dirs=inc_dirs),

        Extension("pysph.base.cell",
                  ["source/pysph/base/cell.pyx"], include_dirs=inc_dirs),

        Extension("pysph.base.polygon_array",
                  ["source/pysph/base/polygon_array.pyx"], 
                  include_dirs=inc_dirs),

        Extension("pysph.base.nnps",
                  ["source/pysph/base/nnps.pyx"], include_dirs=inc_dirs),

        Extension("pysph.base.geometry",
                  ["source/pysph/base/geometry.pyx"], 
                  include_dirs=inc_dirs),
	] 

sph = [
        Extension("pysph.sph.sph_func",
                  ["source/pysph/sph/sph_func.pyx"], include_dirs=inc_dirs,
                  ),

        Extension("pysph.sph.sph_calc",
                  ["source/pysph/sph/sph_calc.pyx"], include_dirs=inc_dirs,
                  ),

        Extension("pysph.sph.kernel_correction",
                  ["source/pysph/sph/kernel_correction.pyx"], 
                  include_dirs=inc_dirs),

        Extension("pysph.sph.funcs.basic_funcs",
                  ["source/pysph/sph/funcs/basic_funcs.pyx"], 
                  include_dirs=inc_dirs),

        Extension("pysph.sph.funcs.position_funcs",
                  ["source/pysph/sph/funcs/position_funcs.pyx"], 
                  include_dirs=inc_dirs),

        Extension("pysph.sph.funcs.boundary_funcs",
                  ["source/pysph/sph/funcs/boundary_funcs.pyx"], 
                  include_dirs=inc_dirs),

        Extension("pysph.sph.funcs.external_force",
                  ["source/pysph/sph/funcs/external_force.pyx"], 
                  include_dirs=inc_dirs),

        Extension("pysph.sph.funcs.density_funcs",
                  ["source/pysph/sph/funcs/density_funcs.pyx"], 
                  include_dirs=inc_dirs),

        Extension("pysph.sph.funcs.energy_funcs",
                  ["source/pysph/sph/funcs/energy_funcs.pyx"], 
                  include_dirs=inc_dirs),

         Extension("pysph.sph.funcs.viscosity_funcs",
                   ["source/pysph/sph/funcs/viscosity_funcs.pyx"], 
                   include_dirs=inc_dirs),

        Extension("pysph.sph.funcs.pressure_funcs",
                  ["source/pysph/sph/funcs/pressure_funcs.pyx"], 
                  include_dirs=inc_dirs),

         Extension("pysph.sph.funcs.xsph_funcs",
                   ["source/pysph/sph/funcs/xsph_funcs.pyx"], 
                   include_dirs=inc_dirs),

    Extension("pysph.sph.funcs.eos_funcs",
              ["source/pysph/sph/funcs/eos_funcs.pyx"], 
              include_dirs=inc_dirs),

    Extension("pysph.sph.funcs.adke_funcs",
              ["source/pysph/sph/funcs/adke_funcs.pyx"], 
              include_dirs=inc_dirs),

        ]

parallel = [
    Extension("pysph.parallel.parallel_controller",
              ["source/pysph/parallel/parallel_controller.pyx"],
              include_dirs=inc_dirs + mpi_inc_dir),

    Extension("pysph.parallel.parallel_cell",
              ["source/pysph/parallel/parallel_cell.pyx"],
              include_dirs=inc_dirs + mpi_inc_dir),

     Extension("pysph.parallel.fast_utils",
               ["source/pysph/parallel/fast_utils.pyx"], include_dirs=inc_dirs),
    ]

# kernel extension modules.
kernels = [Extension("pysph.base.kernels",
                     ["source/pysph/base/kernels.pyx"], include_dirs=inc_dirs),
           ]

solver = [
    Extension("pysph.solver.particle_generator",
              ["source/pysph/solver/particle_generator.pyx"], 
              include_dirs=inc_dirs),
    ]
          

# all extension modules.
ext_modules = base + kernels + sph + solver

if HAS_MPI4PY:
    ext_modules += parallel
    pass

setup(name='PySPH',
      version = '0.9beta',
      author = 'PySPH Developers',
      author_email = 'pysph-dev@googlegroups.com',
      description = "A general purpose Smoothed Particle Hydrodynamics framework",
      long_description = __doc__,
      license = "BSD",
      keywords = "SPH simulation computational fluid dynamics",
      test_suite = "nose.collector",
      packages = find_packages('source'),
      package_dir = {'': 'source'},

      ext_modules = ext_modules,
      
      include_package_data = True,
      cmdclass={'build_ext': build_ext},
      #install_requires=['mpi4py>=1.2', 'numpy>=1.0.3', 'Cython>=0.12'],
      #setup_requires=['Cython>=0.13', 'setuptools>=0.6c1'],
      #extras_require={'3D': 'Mayavi>=3.0'},
      zip_safe = False,
      )

