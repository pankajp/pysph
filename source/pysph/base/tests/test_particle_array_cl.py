""" Tests for the OpenCL functions for ParticleArray """

import pysph.solver.cl_utils as cl_utils
if cl_utils.HAS_CL:
    import pyopencl as cl

import unittest
import numpy

import pysph.base.api as base

class ParticleArrayCLTestCase(unittest.TestCase):
    """ Simple test to assert that creation of OpenCL device buffers works """

    def setUp(self):
        """ Create some default particle properties """

        x = numpy.linspace(0,1,11)
        m = numpy.ones_like(x) * (x[1] - x[0])
        rho = numpy.ones_like(x)
        h = 2*m

        pa = base.get_particle_array(floating_point_default="float",
                                     x=x, h=h, m=m, rho=rho)

        self.pa = pa

        platforms = cl.get_platforms()
        devices = platforms[0].get_devices()

        ctx = cl.Context(devices)
        queue = cl.CommandQueue(ctx, devices[0])

        self.queue = queue

    def test_setup(self):
        """ Test for the creation of default properties and types """

        # default properties as copied from particles.py

        default_props = ['x','y','z','u','v','w','m','h','p','e','rho','cs',
                         'tmpx','tmpy','tmpz']

        nbytes = {'float':32}

        pa = self.pa

        for prop in default_props:
            self.assertTrue( prop in pa.properties )

            # each of these props should be of c_type 'float'

            prop_arr = pa.properties[prop]

            self.assertTrue( prop_arr.get_c_type(), 'float' )

            # assert that the numpy array dtype has the correct size

            npy_array = prop_arr.get_npy_array()

            self.assertTrue( numpy.nbytes[type(npy_array[0])], nbytes['float'])

    def test_create_cl_arrays(self):
        """ Test the creation of the OpenCL arrays """

        pa = self.pa

        # create the OpenCL arrays
        pa.create_cl_arrays(self.queue)

        for prop in pa.properties:

            cl_prop = 'cl_' + prop

            self.assertTrue( pa.cl_properties.has_key(cl_prop) )

            array = pa.cl_properties.get(cl_prop)

            pysph_arr = pa.properties[prop].get_npy_array()

            _array = numpy.empty_like(pysph_arr)

            array.get(self.queue, _array)

            self.assertEqual( len(_array), len(pysph_arr) )

            np = len(_array)

            for i in range(np):
                self.assertEqual( _array[i], pysph_arr[i] )

    def test_create_cl_buffers(self):
        """ Test the `create_cl_buffers` function """

        pa = self.pa
        queue = self.queue
        
        # setup PyOpenCL

        pa.setupCl(self.queue)

        self.assertEqual( pa.context, queue.context )
        self.assertEqual( pa.device, queue.device )

        self.assertTrue( pa.cl_setup_done )

        # create the buffers

        pa.create_cl_buffers()

        # Check the pa and tag buffers on host and device
        np = pa.get_number_of_particles()
        
        pa_buf_host = pa.pa_buf_host
        pa_buf_device = pa.pa_buf_device

        self.assertEqual( numpy.size(pa_buf_host), np )

        host_array = pa.pa_buf_device.get_host_array(pa_buf_host.shape,
                                                     pa_buf_host.dtype)

        self.assertEqual( numpy.size(host_array), np )

        for i in range(np):
            for j in range(16):
                self.assertEqual( host_array[i][j], pa_buf_host[i][j] )

        pa_tag_host = pa.pa_tag_host
        pa_tag_device = pa.pa_tag_device

        host_tag_array = pa_tag_device.get_host_array(pa_tag_host.shape,
                                                      pa_tag_host.dtype)

        self.assertEqual( numpy.size(pa_tag_host), np )
        self.assertEqual( numpy.size(host_tag_array), np )

        for i in range(np):
            for j in range(2):
                self.assertEqual( pa_tag_host[i][j], host_tag_array[i][j] )

if __name__ == '__main__':
    unittest.main()