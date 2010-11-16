"""module to test the timings of various kernel evaluations"""

from pysph.base.point cimport Point, Point_new, Point_add, \
            Point_sub, Point_length
from pysph.base.kernels cimport KernelBase

from time import time

# the kernels to test defined in module pysph.base.kernels
kernel_names = ['Poly6Kernel', 'CubicSplineKernel', 'QuinticSplineKernel',
                'WendlandQuinticSplineKernel', 'HarmonicKernel',
                'GaussianKernel', 'M6SplineKernel', 'W8Kernel', 'W10Kernel']


cdef double r_lo = 0
cdef double r_hi = 5

cdef Point P = Point()
cdef double h = 1.0

cdef long N = 1000
cdef list points = []
for i in range(N):
    points.append(Point())

cpdef dict kernel():
    """kernel function evaluation bench"""
    cdef double t, t2
    cdef dict ret = {}
    cdef double fx, fy, fz
    cdef KernelBase kernel
    cdef int tmp
    cdef long i
    cdef int dim
    for dim from 1<=dim<=3:
        fx = fy = fz = 0
        for tmp from 1 <= tmp <= dim:
            fx = 1/(dim**0.5)
        for i in range(N):
            points[i].x = fx*(r_lo + (r_hi-r_lo)*i/N)
            points[i].y = fx*(r_lo + (r_hi-r_lo)*i/N)
            points[i].z = fx*(r_lo + (r_hi-r_lo)*i/N)
        
        for kernel_name in kernel_names:
            kernel = getattr(kernels, kernel_name)(dim)
            
            t = time()
            for i in range(N):
                val = kernel.function(P, points[i], h)
            t2 = time()
            ret['%dD ' %dim + kernel_name] = (t2-t)/N
    return ret

cpdef dict gradient():
    """kernel gradient evaluation bench"""
    cdef double t, t2
    cdef dict ret = {}
    cdef double fx, fy, fz
    cdef KernelBase kernel
    cdef int tmp
    cdef long i
    cdef int dim
    cdef Point p = Point()
    for dim from 1<=dim<=3:
        fx = fy = fz = 0
        for tmp from 1 <= tmp <= dim:
            fx = 1/(dim**0.5)
        for i in range(N):
            points[i].x = fx*(r_lo + (r_hi-r_lo)*i/N)
            points[i].y = fx*(r_lo + (r_hi-r_lo)*i/N)
            points[i].z = fx*(r_lo + (r_hi-r_lo)*i/N)
        
        for kernel_name in kernel_names:
            kernel = getattr(kernels, kernel_name)(dim)
            
            t = time()
            for i in range(N):
                kernel.gradient(P, points[i], h, p)
            t2 = time()
            ret['%dD ' %dim + kernel_name+' gradient'] = (t2-t)/N
    return ret


cdef list funcs = [kernel, gradient]


cpdef bench():
    """returns a list of a dict of kernel evaluation timings"""
    cdef list timings = []
    for func in funcs:
        timings.append(func())
    return timings # dict of test:time
    
if __name__ == '__main__':
    print bench()
