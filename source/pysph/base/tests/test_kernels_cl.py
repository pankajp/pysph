""" Test for the OpenCL implementation of the SPH smoothing kernels """

import pysph.base.api as base
import pysph.solver.api as solver

import sys
import os
import unittest
import numpy

if solver.HAS_CL:
    import pyopencl as cl
    from pyopencl.array import vec
else:
    sys.exit()

class CubicSplineKernelTestCase(unittest.TestCase):
    """ Tests for the cubic spline kernrel """

    def setUp(self):

        devices = solver.get_cl_devices()

        self.gpu_devices = devices['GPU']
        self.cpu_devices = devices['CPU']

        src_path = os.path.split( os.path.abspath('.'))[0]

        self.prog_src = open(os.path.join(src_path, 'kernels.cl')).read()

        self.cl_inc_dirs = solver.get_cl_include()

        self.gpu_ctx = cl.Context(self.gpu_devices)
        self.cpu_ctx = cl.Context(self.cpu_devices)

        self.mf = cl.mem_flags

    def test_fac(self):

        for dim in range(1,4):
            pysph_kernel = base.CubicSplineKernel( dim=dim )
            pysph_fac = pysph_kernel.py_fac( 1.0 )

            host_h = numpy.ones(shape=(1,), dtype=numpy.float32)
            host_dim = numpy.ones(shape=(1,), dtype=numpy.unsignedinteger)*dim
            host_fac = numpy.zeros(shape=(1,), dtype=numpy.float32)

            mf = self.mf

            ctx = self.gpu_ctx

            for device in self.gpu_devices:
                q = cl.CommandQueue(ctx, device)

                device_h = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                     hostbuf=host_h)

                device_dim = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=host_dim)
            
                device_fac = cl.Buffer(ctx, mf.WRITE_ONLY, host_fac.nbytes)
                
                prog = cl.Program(ctx, self.prog_src).build(
                    options=self.cl_inc_dirs)

                prog.cl_cubic_spline_fac(q, (1,), None, device_h, device_dim,
                                         device_fac)

                cl.enqueue_read_buffer(q, device_fac, host_fac)

                self.assertAlmostEqual(host_fac[0], pysph_fac, 5)
                             
    def test_function(self):

        # test in 1D

        pysph_kernel = base.CubicSplineKernel(dim=1)
        pa = base.Point(0)
        pb = base.Point(1)
        h = 1.0

        pysph_w = pysph_kernel.py_function(pa,pb,h)
        pysph_grad = base.Point()
        pysph_kernel.py_gradient(pa,pb,h,pysph_grad)

        # create the host arrays
        host_pa = numpy.zeros(shape=(1,), dtype=vec.float4)
        host_pb = numpy.zeros(shape=(1,), dtype=vec.float4)
        
        gpu_ctx = self.gpu_ctx
        host_dim = numpy.zeros(shape=(1,), dtype=numpy.unsignedinteger)

        host_w = numpy.zeros(shape=(1,), dtype=numpy.float32)
        host_grad = numpy.zeros(shape=(1,), dtype=vec.float4)

        # set the host array values
        host_pa[0][3] = 1.0
        host_pb[0][0] = 1.0; host_pb[0][3] = 1.0
        host_dim[0] = 1

        # allocate the device arrays

        mf = self.mf
        device_pa = cl.Buffer(gpu_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                              hostbuf=host_pa)

        device_pb = cl.Buffer(gpu_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                              hostbuf=host_pb)

        device_dim = cl.Buffer(gpu_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                               hostbuf=host_dim)

        device_w = cl.Buffer(gpu_ctx, mf.WRITE_ONLY, host_w.nbytes)
        device_grad = cl.Buffer(gpu_ctx, mf.WRITE_ONLY, host_grad.nbytes)

        for gpu in self.gpu_devices:
            q = cl.CommandQueue(gpu_ctx, gpu)

            prog = cl.Program(gpu_ctx,self.prog_src).build(
                options=self.cl_inc_dirs)

            # execute the kernel

            prog.cl_cubic_spline_function(q, (1,), None, device_pa, device_pb,
                                          device_w, device_dim)

            prog.cl_cubic_spline_gradient(q, (1,), None, device_pa, device_pb,
                                          device_grad, device_dim)

            cl.enqueue_read_buffer(q, device_w, host_w)
            cl.enqueue_read_buffer(q, device_grad, host_grad)

            self.assertAlmostEqual(host_w[0], pysph_w, 5)

            self.assertAlmostEqual(host_grad[0][0], pysph_grad.x, 5)
            self.assertAlmostEqual(host_grad[0][1], pysph_grad.y, 5)
            self.assertAlmostEqual(host_grad[0][2], pysph_grad.z, 5)

if __name__ == '__main__':
    unittest.main()
