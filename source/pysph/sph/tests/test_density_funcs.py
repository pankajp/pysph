"""
Tests for the density_funcs module.
"""

# standard imports
import unittest
import numpy
import os

# local imports
import pysph.base.api as base
import pysph.sph.api as sph

from pysph.sph.funcs.density_funcs import SPHRho, SPHDensityRate
from pysph.base.particle_array import ParticleArray
from pysph.base.kernels import Poly6Kernel, CubicSplineKernel

#import opencl if available

HAVE_CL = False
try:
    import pyopencl as cl
    from pyopencl.array import vec
    HAVE_CL = True
except ImportError:
    pass

def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-16."""
    return numpy.allclose(x, y, atol=1e-16, rtol=0)

def get_sample_data1(mass=1.0, radius=1.0):
    """
    Generate sample data for tests in this module.

    A particle array with two points (0, 0, 0) and (1, 1, 1) is created and
    returned. 
    """
    x = [0.0, 1.0]
    y = [0.0, 1.0]
    z = [0.0, 1.0]
    m = [mass, mass]
    h = [radius, radius]
    
    p = ParticleArray(x={'data':x}, y={'data':y}, z={'data':z},
                      m={'data':m}, h={'data':h})
    return p

def get_sample_data2(mass=1.0, radius=1.0):
    """
    Generate sample data for tests in this module.

    Two particle arrays with one point each is returned. The points and values
    at them are the same as in the above function.

    """
    p1 = ParticleArray(x={'data':[0.0]},
                       y={'data':[0.0]},
                       z={'data':[0.0]},
                       m={'data':[mass]},
                       h={'data':[radius]})

    p2 = ParticleArray(x={'data':[1.0]},
                       y={'data':[1.0]},
                       z={'data':[1.0]},
                       m={'data':[mass]},
                       h={'data':[radius]})

    return p1, p2


class CLSummationDenstiyTestCase(unittest.TestCase):
    def setUp(self):
        self.np = np = 101
        self.x = x = numpy.linspace(-1,1,np)
        self.y = y = numpy.zeros_like(x)
        self.z = z = numpy.zeros_like(x)

        self.m = m = numpy.ones_like(x)
        self.h = h = numpy.ones_like(x) * 2 * (x[1] - x[0])

        self.pa = pa = base.get_particle_array(name="test", x=x, y=y, z=z, h=h,
                                               m=m)
        
        self.particles = base.Particles(arrays=[pa,])

        self.kernel = kernel = base.CubicSplineKernel(dim=1)
        func = sph.SPHRho.get_func(pa, pa)

        self.calc = sph.SPHCalc(particles=self.particles,sources=[pa,], dest=pa,
                                funcs=[func,], kernel=kernel, updates=['rho'],
                                integrates=False, dnum=0, nbr_info=True,
                                id="sd")

        # load the kernel source
        path = os.path.abspath('.')
        path = os.path.split(path)[0]
        path = os.path.join(path, '/funcs/density_funcs.cl')
        self.src = open('../funcs/density_funcs.cl').read()

        self.devices = devices = cl.get_platforms()[0].get_devices()
        self.device = device = devices[0]

        self.ctx = ctx = cl.Context(devices)
        self.q = q = cl.CommandQueue(ctx)

        self.mf = mf = cl.mem_flags

        self.prog = prog = cl.Program(ctx, self.src).build()

    def test_point_norm(self):

        host_point = numpy.zeros(shape=(1,), dtype=vec.float4)
        host_norm = numpy.zeros(shape=(1,), dtype=numpy.float32)

        host_point[0][0] = 1.0
        host_point[0][1] = 1.0
        host_point[0][2] = 1.0
        host_point[0][3] = 1.0

        devices = cl.get_platforms()[0].get_devices()
        device = devices[0]

        ctx = cl.Context(devices)
        q = cl.CommandQueue(ctx)

        mf = cl.mem_flags

        buffer_point = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=host_point)
        
        buffer_norm = cl.Buffer(ctx, mf.WRITE_ONLY, host_norm.nbytes)

        prog = cl.Program(ctx, self.src).build()

        prog.test_point_norm(q, (1,), None, buffer_point, buffer_norm)

        cl_norm = numpy.empty(shape=(1,), dtype=numpy.float32)
        cl.enqueue_read_buffer(q, buffer_norm, cl_norm)

        self.assertAlmostEqual(cl_norm[0], numpy.sqrt(3.0), 5)

    def test_kernel(self):

        ctx = self.ctx
        mf = self.mf
        prog = self.prog

        host_pa = numpy.array([vec.make_float4(x=1.0, w=1.0),],
                              dtype=vec.float4)
        
        host_pb = numpy.array([vec.make_float4(x=0.0, w=1.0),],
                              dtype=vec.float4)

        # allocate the buffer objects on the device

        device_pa = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                              hostbuf=host_pa)

        device_pb = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                              hostbuf=host_pb)

        device_g = cl.Buffer(self.ctx, mf.WRITE_ONLY, host_pa.nbytes)

        device_w = cl.Buffer(self.ctx, mf.WRITE_ONLY, host_pa.nbytes/4)


        # test the function evaluation
        
        prog.test_cubic_spline_function(self.q, (1,), None, device_pa,
                                        device_pb, device_w)

        result = numpy.ones(shape=(1,), dtype=numpy.float32) * -1000
        cl.enqueue_read_buffer(self.q, device_w, result)

        # Compare with PySPH result
        
        pa = base.Point(1.)
        pb = base.Point()
        grad = base.Point()

        pysph_function = self.kernel.py_function(pa,pb,1.0)

        self.assertAlmostEqual(result, pysph_function, 5)


        # test the kernel gradient evaluation

        prog.test_cubic_spline_gradient(self.q, (1,), None, device_pa,
                                        device_pb, device_g)

        result = numpy.array([vec.make_float4(),], dtype=vec.float4)
        cl.enqueue_read_buffer(self.q, device_g, result)

        self.kernel.py_gradient(pa, pb, 1.0, grad)

        self.assertAlmostEqual(result[0][0], grad.x, 5)
        self.assertAlmostEqual(result[0][1], grad.y, 5)
        self.assertAlmostEqual(result[0][2], grad.z, 5)

    def test_cl_summation_density(self):
        """ Print testing the OpenCL summation density function """

        np = self.np

        # Perform the refernce PySPH solution
        self.calc.sph()
        pysph_rho = self.pa.tmpx

        # Perform the OpenCL solution

        devices = cl.get_platforms()[0].get_devices()
        device = devices[0]

        ctx = cl.Context(devices)
        q = cl.CommandQueue(ctx)

        mf = cl.mem_flags

        # allocate device buffers

        dst = numpy.zeros(shape=(np,), dtype=vec.float4)
        src = numpy.zeros(shape=(np,), dtype=vec.float4)
        rho = numpy.zeros(shape=(np,), dtype=numpy.float32)

        for i in range(np):
            dst[i][0] = self.x[i]
            dst[i][3] = self.h[i]

            src[i][0] = self.x[i]
            src[i][3] = self.h[i]
                
        dst_buf = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=dst)
        src_buf = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=src)
        rho_buf = cl.Buffer(ctx,mf.WRITE_ONLY, rho.nbytes)

        self.prog.summation_density(q, (101,1,1), (1,1,1),
                                    dst_buf, src_buf, rho_buf)

        cl_rho = numpy.zeros(shape=(np,), dtype=numpy.float32)
        cl.enqueue_read_buffer(q, rho_buf, cl_rho)

        for i in range(np):
            self.assertAlmostEqual(cl_rho[i], pysph_rho[i], 4)

if __name__ == '__main__':
    unittest.main()
