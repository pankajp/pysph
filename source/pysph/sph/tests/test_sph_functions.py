""" test sanity of various functions """

import unittest
import time

from pysph.base.point import Point
from pysph.base import kernels
from pysph.base.carray import DoubleArray
from pysph.base.particle_array import ParticleArray
from pysph.base.api import get_particle_array, Particles
from pysph.sph.sph_calc import SPHCalc
from pysph.base.kernels import KernelBase
from pysph.sph.function import *
import numpy

# the sph_funcs to test
funcs_calc = [
              SPHInterpolation('rho'),
              SPHSimpleGradient('rho'),
              SPHGradient('rho'),
              Laplacian('rho'),
              MonaghanBoundaryForce(delp=1.0),
              BeckerBoundaryForce(sound_speed=1.0),
              LennardJonesForce(D=1.0, ro=1.0, p1=1.0, p2=1.0),
              SPHRho(),
              SPHDensityRate(),
              EnergyEquationNoVisc(),
              EnergyEquationAVisc(beta=1.0, alpha=1.0, gamma=1.0, eta=1.0),
              EnergyEquation(beta=1.0, alpha=1.0, gamma=1.4, eta=0.1),
              ArtificialHeat(),
              PositionStepping(),
              SPHPressureGradient(),
              MomentumEquation(alpha=1.0, beta=1.0, gamma=1.4, eta=0.1),
              MonaghanArtificialVsicosity(alpha=1.0, beta=1.0, gamma=1.4, eta=0.1),
              MorrisViscosity(mu='mu'),
              XSPHCorrection(eps=0.5),
              ADKEPilotRho(),
              VelocityDivergence(),
              XSPHDensityRate(),
             ]


funcs_eqn = [
             PositionStepping(),
             IdealGasEquation(gamma=1.4),
             TaitEquation(co=1.0, ro=1.0, gamma=7.0),
             GravityForce(gx=0.0, gy=-9.81, gz=0.0),
             VectorForce(force=Point(1,1,1)),
             MoveCircleX(),
             MoveCircleY(),
             NeighborCount(),
            ]


Ns = [100]#, 100000]

class TestSPHFuncCalcs(unittest.TestCase):
    pass

class TestSPHFuncEqns(unittest.TestCase):
    pass


# function names have 't' instead of 'test' otherwise nose test collector
# assumes them to be test functions
def create_t_func_calc(func):
    """ create and return test functions for calc sph_funcs """ 
    def test(self):
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
                                    ubar=z, vbar=z, wbar=z)
            pb = get_particle_array(x=x+0.1**0.5, y=y, z=z, h=h, mu=mu, rho=rho, m=m, tmp=z,
                                    tx=m, ty=z, tz=z, nx=z, ny=m, nz=z, u=z, v=z, w=z,
                                    ubar=z, vbar=z, wbar=z)
            particles = Particles(arrays=[pa, pb])
            for func_getter in funcs_calc:
                func = func_getter.get_func(pa, pb)
                calc = SPHCalc(particles, [pa], pb, kernel, [func], ['tmp'])
                t = get_time()
                calc.sph('tmp')
                t = get_time() - t
                nam = '%s:%s'%(func_getter.__class__.__name__, func.__class__.__name__)
                ret[nam +' /%d'%(N)] = t/N
        return ret

    test.__name__ = 'test_sph_func_calc_%s'%(func.__class__.__name__)
    test.__doc__ = 'run sanity check for calc: %s:%s'%(func.__class__.__name__,
                                func.sph_func.__name__)
    
    return test

def create_t_func_eqn(func):
    """ create and return test functions for calc sph_funcs """ 
    def test(self):
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
            pa = get_particle_array(x=x, y=y, z=z, h=h, mu=mu, rho=rho, m=m, tmp=z)
            pb = get_particle_array(x=x+0.1**0.5, y=y, z=z, h=h, mu=mu, rho=rho, m=m, tmp=z)
            particles = Particles(arrays=[pa, pb])
            for func_getter in funcs_eqn:
                func = func_getter.get_func(pa, pb)
                calc = SPHCalc(particles, [pa], pb, kernel, [func], ['tmp'])
                t = get_time()
                calc.sph('tmp')
                t = get_time() - t
                nam = '%s:%s'%(func_getter.__class__.__name__, func.__class__.__name__)
                ret[nam +' /%d'%(N)] = t/N
        return ret

    test.__name__ = 'test_sph_func_eqn_%s'%(func.__class__.__name__)
    test.__doc__ = 'run sanity check for eqn: %s:%s'%(func.__class__.__name__,
                                func.sph_func.__name__)
    
    return test


def gen_ts():
    """ generate test functions and attach them to test classes """
    for i, func in enumerate(funcs_calc):
        t_method = create_t_func_calc(func)
        t_method.__name__ = t_method.__name__ + '_%d'%(i)
        setattr(TestSPHFuncCalcs, t_method.__name__, t_method)

    for i, func in enumerate(funcs_eqn):
        t_method = create_t_func_eqn(func)
        t_method.__name__ = t_method.__name__ + '_%d'%(i)
        setattr(TestSPHFuncEqns, t_method.__name__, t_method)

# generate the test functions
gen_ts()
    
if __name__ == "__main__":
    unittest.main()
