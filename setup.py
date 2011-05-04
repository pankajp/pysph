"""
PySPH
=====

A general purpose Smoothed Particle Hydrodynamics framework.

This package provides a general purpose framework for SPH simulations
in Python.  The framework emphasizes flexibility and efficiency while
allowing most of the user code to be written in pure Python.  See here:

    http://pysph.googlecode.com

for more information.
"""

from setuptools import find_packages, setup
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from numpy.distutils.extension import Extension

import numpy
import sys
import multiprocessing
ncpu = multiprocessing.cpu_count()

inc_dirs = [numpy.get_include()]
extra_compile_args = []
extra_link_args = []

mpi_inc_dirs = []
mpi_compile_args = []
mpi_link_args = []

USE_CPP = True
HAS_MPI4PY = True
try:
    import mpi4py
    # assume a working mpi environment
    import commands
    if USE_CPP:
        mpic = 'mpicxx'
    else:
        mpic = 'mpicc'
    mpi_link_args.append(commands.getoutput(mpic + ' --showme:link'))
    mpi_compile_args.append(commands.getoutput(mpic +' --showme:compile'))
    mpi_inc_dirs.append(mpi4py.get_include())
except ImportError:
    HAS_MPI4PY = False

cy_directives = {'embedsignature':True,
                 }

# base extension modules.
base = [Extension("pysph.base.carray", 
                  ["source/pysph/base/carray.pyx"],),

        Extension("pysph.base.point",
                  ["source/pysph/base/point.pyx"],),

        Extension("pysph.base.plane",
                  ["source/pysph/base/plane.pyx"],),

        Extension("pysph.base.particle_array",
                  ["source/pysph/base/particle_array.pyx"],),

        Extension("pysph.base.cell",
                  ["source/pysph/base/cell.pyx"],),

        Extension("pysph.base.polygon_array",
                  ["source/pysph/base/polygon_array.pyx"],),

        Extension("pysph.base.nnps",
                  ["source/pysph/base/nnps.pyx"],),

        Extension("pysph.base.geometry",
                  ["source/pysph/base/geometry.pyx"],),
	] 

sph = [
        Extension("pysph.sph.sph_func",
                  ["source/pysph/sph/sph_func.pyx"],),

        Extension("pysph.sph.sph_calc",
                  ["source/pysph/sph/sph_calc.pyx"],),

        Extension("pysph.sph.kernel_correction",
                  ["source/pysph/sph/kernel_correction.pyx"],),

        Extension("pysph.sph.funcs.basic_funcs",
                  ["source/pysph/sph/funcs/basic_funcs.pyx"],),

        Extension("pysph.sph.funcs.position_funcs",
                  ["source/pysph/sph/funcs/position_funcs.pyx"],),

        Extension("pysph.sph.funcs.boundary_funcs",
                  ["source/pysph/sph/funcs/boundary_funcs.pyx"],),

        Extension("pysph.sph.funcs.external_force",
                  ["source/pysph/sph/funcs/external_force.pyx"],),

        Extension("pysph.sph.funcs.density_funcs",
                  ["source/pysph/sph/funcs/density_funcs.pyx"],),

        Extension("pysph.sph.funcs.energy_funcs",
                  ["source/pysph/sph/funcs/energy_funcs.pyx"],),

        Extension("pysph.sph.funcs.viscosity_funcs",
                   ["source/pysph/sph/funcs/viscosity_funcs.pyx"],),

        Extension("pysph.sph.funcs.pressure_funcs",
                  ["source/pysph/sph/funcs/pressure_funcs.pyx"],),

        Extension("pysph.sph.funcs.xsph_funcs",
                  ["source/pysph/sph/funcs/xsph_funcs.pyx"],),

        Extension("pysph.sph.funcs.eos_funcs",
                  ["source/pysph/sph/funcs/eos_funcs.pyx"],),

        Extension("pysph.sph.funcs.adke_funcs",
                  ["source/pysph/sph/funcs/adke_funcs.pyx"],),

        Extension("pysph.sph.funcs.arithmetic_funcs",
                  ["source/pysph/sph/funcs/arithmetic_funcs.pyx"],),

        ]

parallel = [
        Extension("pysph.parallel.parallel_controller",
                  ["source/pysph/parallel/parallel_controller.pyx"],),

        Extension("pysph.parallel.parallel_cell",
                  ["source/pysph/parallel/parallel_cell.pyx"],),

        Extension("pysph.parallel.fast_utils",
                  ["source/pysph/parallel/fast_utils.pyx"],),
        ]

# kernel extension modules.
kernels = [Extension("pysph.base.kernels",
                     ["source/pysph/base/kernels.pyx"],),
           ]

solver = [
          Extension("pysph.solver.particle_generator",
                    ["source/pysph/solver/particle_generator.pyx"],),
          ]


# all extension modules.
ext_modules = base + kernels + sph + solver

if HAS_MPI4PY:
    ext_modules += parallel

for extn in ext_modules:
    extn.include_dirs = inc_dirs
    extn.extra_compile_args = extra_compile_args
    extn.extra_link_args = extra_link_args
    extn.pyrex_directives = cy_directives
    if USE_CPP:
        extn.language = 'c++'

for extn in parallel:
    extn.include_dirs.extend(mpi_inc_dirs)
    extn.extra_compile_args.extend(mpi_compile_args)
    extn.extra_link_args.extend(mpi_link_args)

if 'build_ext' in sys.argv or 'develop' in sys.argv or 'install' in sys.argv:
    ext_modules = cythonize(ext_modules, nthreads=ncpu, include_path=inc_dirs)

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
      #install_requires=[python>=2.6<3', 'mpi4py>=1.2', 'numpy>=1.0.3', 'Cython>=0.14'],
      #setup_requires=['Cython>=0.14', 'setuptools>=0.6c1'],
      #extras_require={'3D': 'Mayavi>=3.0'},
      zip_safe = False,
      )

