"""A general purpose Smoothed Particle Hydrodynamics framework.

The SPH package provides a general purpose framework for SPH simulations
in Python.  The framework emphasizes flexibility and efficiency while
allowing most of the user code to be written in pure Python.
"""

from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
import numpy

HAS_MPI4PY = True
try:
    import mpi4py
except ImportError:
    HAS_MPI4PY = False

inc_dirs = [numpy.get_include()]
if HAS_MPI4PY:
    inc_dirs += [mpi4py.get_include()]

mpi_inc_dir = ['/usr/include/mpi', '/usr/local/include/mpi', 
               '/opt/local/include/mpi']

# base extension modules.
base = [Extension("pysph.base.carray", 
                  ["source/pysph/base/carray.pyx"], include_dirs=inc_dirs),

        Extension("pysph.base.point",
                  ["source/pysph/base/point.pyx"], include_dirs=inc_dirs),
        
        Extension("pysph.base.particle_array",
                  ["source/pysph/base/particle_array.pyx"], include_dirs=inc_dirs),
        
        Extension("pysph.base.nnps",
                  ["source/pysph/base/nnps.pyx"], include_dirs=inc_dirs),
        
        Extension("pysph.base.kernels",
                  ["source/pysph/base/kernels.pyx"]),

        Extension("pysph.base.particle_tags",
                  ["source/pysph/base/particle_tags.pyx"]),
        
	] 

sph = [
        Extension("pysph.sph.sph_func",
                  ["source/pysph/sph/sph_func.pyx"], include_dirs=inc_dirs),

        Extension("pysph.sph.calc",
                  ["source/pysph/sph/calc.pyx"], include_dirs=inc_dirs),

        Extension("pysph.sph.misc_particle_funcs",
                  ["source/pysph/sph/misc_particle_funcs.pyx"], include_dirs=inc_dirs),

        Extension("pysph.sph.funcs",
                  ["source/pysph/sph/funcs.pyx"], include_dirs=[inc_dirs]),

        ]


'''
parallel = [
    Extension("pysph.parallel.parallel_controller",
              ["source/pysph/parallel/parallel_controller.pyx"],
              include_dirs=inc_dirs + mpi_inc_dir),
    Extension("pysph.parallel.cy_parallel_cell",
              ["source/pysph/parallel/cy_parallel_cell.pyx"],
              include_dirs=inc_dirs + mpi_inc_dir),
    ]
'''

parallel = []

solver = [
          Extension("pysph.solver.entity_types",
                    ["source/pysph/solver/entity_types.pyx"], include_dirs=inc_dirs),
          
          Extension("pysph.solver.entity_base",
                    ["source/pysph/solver/entity_base.pyx"], include_dirs=inc_dirs),
          
          Extension("pysph.solver.fluid",
                    ["source/pysph/solver/fluid.pyx"], include_dirs=inc_dirs),
          
          
          Extension("pysph.solver.sph_component",
                    ["source/pysph/solver/sph_component.pyx"], include_dirs=inc_dirs),
          
          Extension("pysph.solver.func_components",
                  ["source/pysph/solver/func_components.pyx"], include_dirs=[inc_dirs]),
          ]

# all extension modules.
ext_modules = base +\
    sph+\
    solver 

if HAS_MPI4PY:
    ext_modules += parallel

setup(name='PySPH',
      version = '0.9',
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
      #setup_requires=['Cython>=0.12', 'setuptools>=0.6c1'],
      #extras_require={'3D': 'Mayavi>=3.0'},
      zip_safe = False,
      )

