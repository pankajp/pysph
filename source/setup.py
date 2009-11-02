"""A general purpose Smoothed Particle Hydrodynamics framework.

The SPH package provides a general purpose framework for SPH simulations
in Python.  The framework emphasizes flexibility and efficiency while
allowing most of the user code to be written in pure Python.
"""

from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext

# base extension modules.
base = [Extension("pysph.base.carray", 
                  ["pysph/base/carray.pyx"]),
        Extension("pysph.base.point",
                  ["pysph/base/point.pyx"]),
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

solver = [Extension("pysph.solver.typed_dict",
                    ["pysph/solver/typed_dict.pyx"]),
          Extension("pysph.solver.base",
                    ["pysph/solver/base.pyx"]),
          Extension("pysph.solver.entity_base",
                    ["pysph/solver/entity_base.pyx"]),
          Extension("pysph.solver.entity_types",
                    ["pysph/solver/entity_types.pyx"]),
          Extension("pysph.solver.solver_component",
                    ["pysph/solver/solver_component.pyx"]),
          ]

# all extension modules.
ext_modules = base +\
    kernels +\
    sph+\
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
