"""This module compiles the specified (all) the cython .pyx files
in the specified (current) directory into python extensions
"""

import sys
import os

from setuptools import setup
from Cython.Distutils import build_ext
from numpy.distutils.extension import Extension

import numpy

def get_spcl_extn(extn):
    """ special-case extensions with specific requirements """
    if extn.name == 'cpp_vs_pyx':
        extn.language = 'c++'
        extn.sources.append('cPoint.cpp')
    return extn

def compile_extns(extensions=None, dirname=None, inc_dirs=None):
    """compile cython extensions
    
    `extensions` is list of extensions to compile (None => all pyx files)
    `dirname` is directory in which extensions are found (None = current directory)
    `inc_dirs` is list of additional cython include directories
    """
    if dirname is None:
        dirname = os.path.abspath(os.curdir)
    olddir = os.path.abspath(os.curdir)
    os.chdir(dirname)
    
    if extensions is None:
        extensions = sorted([f[:-4] for f in os.listdir(os.curdir) if f.endswith('.pyx')])
    
    if inc_dirs is None:
        inc_dirs = []
    inc_dirs.append(os.path.join(os.path.split(os.path.abspath(os.path.curdir))[0],'source'))
    print inc_dirs
    sys.argvold = sys.argv[:]
    sys.argv = [__file__, 'build_ext','--inplace']
    
    inc_dirs = [numpy.get_include()] + inc_dirs
    
    cargs = []#'-O3']
    
    # extension modules
    extns = []
    for extnname in extensions:
        extn = Extension(extnname, [extnname+".pyx"], include_dirs=inc_dirs,
                             extra_compile_args=cargs)
        extn = get_spcl_extn(extn)
        extns.append(extn)
    
    setup(name='PySPH-bench',
          ext_modules = extns,
          include_package_data = True,
          cmdclass={'build_ext': build_ext},
          )
    
    os.chdir(olddir)
    sys.argv = sys.argvold

if __name__ == '__main__':
    if '-h' in sys.argv or '--help' in sys.argv:
        print '''usage:
        python setup.py [extension1, [extension2, [...]]]
        
        compiles the cython extensions present in the current directory
        '''
    elif len(sys.argv) > 1:
        # compile specified extensions
        compile_extns(sys.argv[1:])
    else:
        # compile all extensions found in current directory
        compile_extns()
    
