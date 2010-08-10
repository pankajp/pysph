"""this module compiles all the .pyx files in the directory where this
module file is present into python extensions"""


import sys

sys.argvold = sys.argv[:]
if len(sys.argv) == 1:
    sys.argv.extend(['build_ext','--inplace'])


from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
import numpy

import os

dirname = os.path.abspath(os.path.dirname(__file__))
olddir = os.path.abspath(os.curdir)
os.chdir(dirname)

extensions = sorted([f[:-4] for f in os.listdir(os.curdir) if f.endswith('.pyx')])

inc_dirs = [numpy.get_include(), os.path.join('..','..')]

cargs = ['-O3']

# extension modules
extns = [Extension(extnname, 
                  [extnname+".pyx"], include_dirs=inc_dirs,
                  extra_compile_args=cargs)
        for extnname in extensions
        ]

setup(name='PySPH-bench',
      ext_modules = extns,
      include_package_data = True,
      cmdclass={'build_ext': build_ext},
      )

os.chdir(olddir)
sys.argv = sys.argvold

