""" test sanity of various functions """

import unittest
import time
import numpy

from pysph.base import kernels
from pysph.base.carray import DoubleArray
from pysph.base.particle_array import ParticleArray
from pysph.base.api import get_particle_array, Particles
from pysph.sph.sph_calc import SPHCalc
from pysph.sph.sph_func import get_all_funcs


funcs = get_all_funcs()

Ns = [100]#, 100000]

class TestSPHFuncs(unittest.TestCase):
    pass


# function names have 't' instead of 'test' otherwise nose test collector
# assumes them to be test functions
def create_t_func(func_getter):
    """ create and return test functions for sph_funcs """
    cls = func_getter.get_func_class()
    
    def t(self):
        ret = {}
        da = DoubleArray()
        pa = ParticleArray()
        kernel = kernels.CubicSplineKernel(3)
        get_time = time.time
        for N in Ns:
            x = numpy.arange(N)
            z = y = numpy.zeros(N)
            mu = m = rho = numpy.ones(N)
            h = 2*m
            da = DoubleArray(N)
            da2 = DoubleArray(N)
            da.set_data(z)
            da2.set_data(z)
            pa = get_particle_array(x=x, y=y, z=z, h=h, mu=mu, rho=rho, m=m, tmp=z,
                                    tx=z, ty=m, tz=z, nx=m, ny=z, nz=z, u=z, v=z, w=z,
                                    ubar=z, vbar=z, wbar=z, q=m)
            pb = get_particle_array(x=x+0.1**0.5, y=y, z=z, h=h, mu=mu, rho=rho, m=m, tmp=z,
                                    tx=m, ty=z, tz=z, nx=z, ny=m, nz=z, u=z, v=z, w=z,
                                    ubar=z, vbar=z, wbar=z, q=m)
            particles = Particles(arrays=[pa, pb])
            
            func = func_getter.get_func(pa, pb)
            calc = SPHCalc(particles, [pa], pb, kernel, [func], ['tmp'])
            print cls.__name__
            t = get_time()
            calc.sph('tmp', 'tmp', 'tmp')
            t = get_time() - t
            
            nam = '%s'%(cls.__name__)
            ret[nam +' /%d'%(N)] = t/N
        return ret

    t.__name__ = 'test_sph_func__%s'%(cls.__name__)
    t.__doc__ = 'run sanity check for calc: %s'%(cls.__name__)
    
    return t


def gen_ts():
    """ generate test functions and attach them to test classes """
    for i, func in enumerate(funcs.values()):
        t_method = create_t_func(func)
        t_method.__name__ = t_method.__name__ + '_%d'%(i)
        setattr(TestSPHFuncs, t_method.__name__, t_method)

# generate the test functions
gen_ts()
    
if __name__ == "__main__":
    unittest.main()
