"""
Tests for the density_funcs module.
"""

# standard imports
import unittest
import numpy
import os

#import opencl if available
import pysph.solver.cl_utils as cl_utils
import pyopencl as cl

# local imports
import pysph.base.api as base
import pysph.sph.api as sph

from pysph.sph.funcs.density_funcs import SPHRho, SPHDensityRate
from pysph.base.particle_array import ParticleArray
from pysph.base.kernels import Poly6Kernel, CubicSplineKernel

def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-16."""
    return numpy.allclose(x, y, atol=1e-16, rtol=0)

class CLSPHRhoTestCase(unittest.TestCase):
    def setUp(self):
        self.np = np = 11
        self.x = x = numpy.linspace(-1,1,np).astype(numpy.float32)
        self.y = y = numpy.zeros_like(x)
        self.z = z = numpy.zeros_like(x)

        self.m = m = numpy.ones_like(x)
        self.h = h = numpy.ones_like(x) * 2 * (x[1] - x[0])

        self.pa = pa = base.get_particle_array(name="test", x=x, y=y, z=z, h=h,
                                               m=m)
        
        self.func = func = sph.SPHRho.get_func(pa, pa)

        # load the kernel source
        root = cl_utils.get_pysph_root()
        src = open(os.path.join(root, 'sph/funcs/density_funcs.cl')).read()

        self.devices = devices = cl.get_platforms()[0].get_devices()
        self.device = device = devices[0]

        self.ctx = ctx = cl.Context(devices)
        self.q = q = cl.CommandQueue(ctx)

        self.prog = prog = cl.Program(
            ctx,src).build(cl_utils.get_cl_include())

        kernel = base.CubicSplineKernel(dim=1)

        self.kernel_type = numpy.array([1,], numpy.int32)
        self.dim = numpy.array([1,], numpy.int32)
        self.nbrs = numpy.array([np,], numpy.int32)

    def test_SPHRho(self):
        """ Print testing the OpenCL summation density function SPHRho"""

        pa = self.pa
        np = pa.get_number_of_particles()

        # Perform the OpenCL solution

        mf = cl.mem_flags

        pa.setupCL(self.ctx)

        xbuf = pa.get_cl_buffer('x')
        ybuf = pa.get_cl_buffer('y')
        zbuf = pa.get_cl_buffer('z')
        hbuf = pa.get_cl_buffer('h')
        mbuf = pa.get_cl_buffer('m')
        rhobuf = pa.get_cl_buffer('rho')
        tagbuf = pa.get_cl_buffer('tag')
        tmpbuf = pa.get_cl_buffer('tmpx')

        kernel_type_buf=cl.Buffer(self.ctx,
                                  mf.READ_WRITE | mf.COPY_HOST_PTR,
                                  hostbuf=self.kernel_type)

        dimbuf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                           hostbuf=self.dim)

        nbrbuf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                           hostbuf=self.nbrs)

        q = self.q
        self.prog.SPHRho(q, (np,1,1), (1,1,1), kernel_type_buf, dimbuf, nbrbuf,
                         xbuf, ybuf, zbuf, hbuf, tagbuf, xbuf, ybuf, zbuf,
                         hbuf, mbuf, rhobuf, tmpbuf)

        tmpx = self.pa.get('tmpx')
        print tmpx

        cl.enqueue_read_buffer(self.q, tmpbuf, tmpx)

        print tmpx
        
if __name__ == '__main__':
    unittest.main()
