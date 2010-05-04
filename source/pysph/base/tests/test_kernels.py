""" Tests for the SPH kernels defined in kernels.pyx
"""

#Author: Kunal Puri <kunalp@aero.iitb.ac.in>
#Copyright (c) 2010, Kunal Puri

import unittest, numpy
import scipy.integrate as integrate

#Local imports
import pysph.base.api as base

#Check the 3D gradients test more thoroughly!

##############################################################################
#`CubicSplineTestCase`
##############################################################################
class CubicSplineTestCase(unittest.TestCase):
    """ Base class for testing Kernels """

    def setUp(self):
        #Set the kernel 
        self.kernel = base.CubicSplineKernel()

        #Set the limits for the integration
        self.xa = -1
        self.xb = 1
        self.ha = lambda x: -1
        self.hb = lambda x: 1
        self.qa = lambda x,y: -1
        self.qb = lambda x,y: 1

        #Default smoothing length
        self.h = 0.01
###############################################################################
        
###############################################################################
#`TestCubicSplineFunction`
###############################################################################
class TestCubicSplineFunction(CubicSplineTestCase):
    """ Test function evaluation in multiple dimensions """

    def test_fac(self):
        dims = 1,2,3
        h = 0.01
        facs = 2./3/h, 10/(7*numpy.pi)/(h*h), 1./(numpy.pi)/(h*h*h)
        
        for i in range(3):
            dim = dims[i]
            fac = facs[i]
            kernel = base.CubicSplineKernel(dim)
            self.assertEqual(kernel.dim, dim)
            self.assertAlmostEqual(kernel.py_fac(h), fac, 6)            
        
    def test_value(self):
        """ Test the function when q = abs(ra - rb) >= 2h. """
        dims = [1, 2, 3]
        h = 0.01
        kernel = self.kernel

        #Check for dimensional compatibility
        p1 = {'x':2, 'y':2}
        self.assertRaises(AssertionError, kernel._function, 2, 2)

        # 1D 
        kernel.dim = dims[0]
        self.assertAlmostEqual(kernel._function(x = 2.1*h), 0.0, 10)

        # 2D
        kernel.dim = dims[1]
        self.assertAlmostEqual(kernel._function(x = 0.0, y = 2.1*h), 0.0, 10)
        self.assertAlmostEqual(kernel._function(x = h, y = 2*h), 0.0, 10)

        #3D
        kernel.dim = dims[2]
        self.assertAlmostEqual(kernel._function(x = h, y = h, 
                                                z = 2*h), 0.0, 10)

    def test_function1D(self):
        """ Test for normalization of the kernel """
        kernel = self.kernel
        kernel.dim = 1
        val, res = integrate.quad(kernel._function, -1, 1)
        self.assertAlmostEqual(val, 1.0, 10)

    def test_function2D(self):
        kernel = self.kernel
        kernel.dim = 2
        val, res = integrate.dblquad(kernel._function, -1, 1, 
                                     self.ha, self.hb)

        self.assertAlmostEqual(val, 1.0, 8)

    def _test_function3D(self):
        print 'Testing 3D, this takes a while!'
        kernel = self.kernel
        kernel.dim = 3
        val, res = integrate.tplquad(kernel._function, -1, 1, 
                                     self.ha, self.hb, self.qa, self.qb)

        self.assertAlmostEqual(val, 1.0, 8)

    def test_gradient1D(self):
        """ Test for the normalization of the kernel in 1D. """
        kernel = self.kernel
        res = base.Point()
        kernel.dim = 1
        val, res = integrate.quad(kernel._gradient1D, -1, 1, args=(res,))
        self.assertAlmostEqual(val, 0.0, 10)
        
    def test_gradient2D(self):
        kernel = self.kernel 
        kernel.dim = 2
        
        #Check for the first component
        res = base.Point()
        val, res = integrate.dblquad(kernel._gradient2D, -1, 1, self.ha,
                                     self.hb, args = (res,))
        self.assertAlmostEqual(val, 0.0, 10)

    def test_gradient3D(self):
        print 'Testing 3D Gradients'
        kernel = self.kernel 
        kernel.dim = 3
        
        #Check for the first component
        res = base.Point()
        val, res = integrate.tplquad(kernel._gradient3D, -1, 1, self.ha,
                                     self.hb, self.qa, self.qb, 
                                     args = (res,))
        self.assertAlmostEqual(val, 0.0, 10)

###############################################################################

if __name__ == '__main__':
    unittest.main()
