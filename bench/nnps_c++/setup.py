from setuptools import find_packages, setup
from Cython.Distutils import build_ext
from numpy.distutils.extension import Extension

ext_modules = [Extension("cython_nnps", ["cython_nnps.pyx"],
                         language="c++",
                         extra_compile_args=["-O3", "-Wall"]
                         ),
               
               Extension("nnps_bench", ["nnps_bench.pyx"],
                         language="c++",
                         extra_compile_args=["-O3", "-Wall"]
                         ),
               ]


setup(
    name = "Cython NNPS",
    cmdclass = {'build_ext':build_ext},
    ext_modules=ext_modules
    )
    
