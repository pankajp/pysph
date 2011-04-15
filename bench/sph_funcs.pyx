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
#from pysph.sph.function import *
from pysph.sph.sph_func import get_all_funcs
from pysph.sph.sph_func cimport SPHFunctionParticle
import numpy

import time

funcs_all = get_all_funcs()

# the sph_funcs to test
funcs_calc = []
funcs_eqn = []
for fname,func in funcs_all.iteritems():
    if fname == 'pysph.sph.funcs.basic_funcs.FirstOrderCorrectionTermAlpha':
        continue
    if issubclass(func.get_func_class(), SPHFunctionParticle):
        funcs_calc.append(func)
    else:
        funcs_eqn.append(func)

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
                                tx=z, ty=m, tz=z, nx=m, ny=z, nz=z, u=z, v=z, w=z,
                                ubar=z, vbar=z, wbar=z, q=m)
        pb = get_particle_array(x=x+0.1**0.5, y=y, z=z, h=h, mu=mu, rho=rho, m=m, tmp=z,
                                tx=m, ty=z, tz=z, nx=z, ny=m, nz=z, u=z, v=z, w=z,
                                ubar=z, vbar=z, wbar=z, q=m)
        particles = Particles(arrays=[pa, pb])
        for func_getter in funcs_calc:
            func = func_getter.get_func(pa, pb)
            calc = SPHCalc(particles, [pa], pb, kernel, [func], ['tmp'])
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
        pa = get_particle_array(x=x, y=y, z=z, h=h, mu=mu, rho=rho, m=m, tmp=z,
                                tx=z, ty=m, tz=z, nx=m, ny=z, nz=z, u=z, v=z, w=z,
                                ubar=z, vbar=z, wbar=z)
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
