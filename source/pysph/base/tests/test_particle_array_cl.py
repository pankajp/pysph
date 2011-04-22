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

        self.ctx = ctx = cl.Context(devices)
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

    def test_create_cl_buffers(self):
        """ Test the creation of the OpenCL arrays """

        pa = self.pa

        # create the OpenCL buffers
        pa.setup_cl(self.ctx, self.queue)

        for prop in pa.properties:

            cl_prop = 'cl_' + prop

            self.assertTrue( pa.cl_properties.has_key(cl_prop) )

            # get the OpenCL buffer for the property
            buffer = pa.get_cl_buffer(prop)

            # get the PySPH numpy array for the property
            pysph_arr = pa.get(prop)

            # read the contents of the OpenCL buffer in a dummy array
            _array = numpy.ones_like(pysph_arr)

            carray = pa.properties[prop]
            dtype = carray.get_c_type()
            if dtype == "double":
                _array = _array.astype(numpy.float32)
            if dtype == "long":
                _array = _array.astype(numpy.int32)

            cl.enqueue_read_buffer(self.queue, buffer, _array).wait()

            self.assertEqual( len(_array), len(pysph_arr) )

            np = len(_array)

            for i in range(np):
                self.assertEqual( _array[i], pysph_arr[i] )

    def test_read_from_buffer(self):

        pa = self.pa

        # create the OpenCL buffers
        pa.setup_cl(self.ctx, self.queue)

        copies = {}
        for prop in pa.properties:
            copies[prop] = pa.get(prop).copy()

        # read the contents back into the array

        pa.read_from_buffer()

        for prop in pa.properties:

            buffer_array = pa.get(prop)
            orig_array = copies.get(prop)

            self.assertEqual( len(buffer_array), len(orig_array) )

            for i in range( len(buffer_array) ):
                self.assertAlmostEqual(buffer_array[i], orig_array[i], 10 )

if __name__ == '__main__':
    unittest.main()
