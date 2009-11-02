"""A general purpose Smoothed Particle Hydrodynamics framework.

The SPH package provides a general purpose framework for SPH simulations
in Python.  The framework emphasizes flexibility and efficiency while
allowing most of the user code to be written in pure Python.
"""

from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext

setup(name='CArrays',
      version = '0.1',
      author = 'Prabhu Ramachandran', # Temporary place holder
      author_email = 'prabhu@aero.iitb.ac.in',
      description = "A general purpose Smoothed Particle Hydrodynamics framework",
      long_description = __doc__,
      license = "Undecided",
      keywords = "SPH simulation computational fluid dynamics",
      test_suite = "nose.collector",

      packages = find_packages(),
      ext_modules = [Extension("cytest.carray",
                               ["cytest/carray.pyx"]),
                     ],
      include_package_data = True,
      cmdclass={'build_ext': build_ext},
    )
