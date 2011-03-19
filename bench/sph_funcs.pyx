"""module to test the timings of various sph_func evaluations"""

from pysph.base.point cimport Point, Point_new, Point_add, Point_sub, \
        Point_length
from pysph.base import kernels
from pysph.base.carray cimport DoubleArray
from pysph.base.particle_array cimport ParticleArray
from pysph.base.api import get_particle_array, Particles
from pysph.sph.sph_calc import SPHCalc
from pysph.sph.sph_calc cimport SPHCalc
from pysph.base.kernels cimport KernelBase
from pysph.sph.function import *
import numpy

import time

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


Ns = [1000]#, 100000]

cpdef dict sph_func_calc(Ns=Ns):
    """sph function function evaluation bench"""
    cdef double t, t2
    cdef dict ret = {}
    cdef double fx, fy, fz
    cdef DoubleArray da
    cdef ParticleArray pa
    cdef int tmp
    cdef long i
    cdef int dim
    cdef KernelBase kernel = kernels.CubicSplineKernel(3)
    cdef SPHCalc calc
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
                                tx=z, ty=z, tz=z, nx=z, ny=z, nz=z, u=z, v=z, w=z,
                                ubar=z, vbar=z, wbar=z)
        particles = Particles(arrays=[pa])
        for func_getter in funcs_calc:
            func = func_getter.get_func(pa, pa)
            calc = SPHCalc(particles, [pa], pa, kernel, [func], ['tmp'])
            t = get_time()
            calc.sph('tmp')
            t = get_time() - t
            nam = func.__class__.__name__+' /%d'%(N)
            if nam in ret:
                print 'error:',nam, ' already in ret:', calc
                nam += 't'
            ret[nam] = t/N
    return ret


cpdef dict sph_func_eqn(Ns=Ns):
    """sph function function evaluation bench"""
    cdef double t, t2
    cdef dict ret = {}
    cdef double fx, fy, fz
    cdef DoubleArray da
    cdef ParticleArray pa
    cdef int tmp
    cdef long i
    cdef int dim
    cdef KernelBase kernel = kernels.CubicSplineKernel(3)
    cdef SPHCalc calc
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
        particles = Particles(arrays=[pa])
        for func_getter in funcs_eqn:
            func = func_getter.get_func(pa, pa)
            calc = SPHCalc(particles, [pa], pa, kernel, [func], ['tmp'])
            t = get_time()
            calc.sph('tmp')
            t = get_time() - t
            ret[func.__class__.__name__+' /%d'%(N)] = t/N
    return ret


cdef list funcs = [sph_func_calc, sph_func_eqn]


cpdef bench():
    """returns a list of a dict of kernel evaluation timings"""
    cdef list timings = []
    for func in funcs:
        timings.append(func())
    return timings # dict of test:time
    
if __name__ == '__main__':
    print bench()
