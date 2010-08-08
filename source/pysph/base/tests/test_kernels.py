""" Tests for the SPH kernels defined in kernels.pyx
"""

#Author: Kunal Puri <kunalp@aero.iitb.ac.in>
#Copyright (c) 2010, Prabhu Ramachandran

import unittest, numpy
import scipy.integrate as integrate

#Local imports
from pysph.base import kernels
from pysph.base.point import Point

# TODO: Check the 3D gradients test more thoroughly
# TODO: Write test for Poly6Kernel

##############################################################################
#`KernelTestCase1D`
##############################################################################
class KernelTestCase1D(unittest.TestCase):
    """ Base class for testing Kernels """

    def setUp(self):
        """ Basic setup for testing the kernel.
        
        Setup:
        ------
        The kernel is assumed to be placed at the origon.
        Particle spacing dx = 0.1
        
        Expected Result:
        ----------------
        The kernel and it's gradient is integrated (summation) 
        over the domain, ensuring no boundary truncation. 
        The result should be 1 and 0 respectively.
        
        """

        #Set the limits for Summation.
        self.x = x = numpy.linspace(-1, 1, 21)
        self.y = numpy.linspace(-1, 1, 21)
        self.dx = dx = x[1] - x[0]
        self.hs = 0.2

        self.pnt = Point()
        self.pnts = [Point(i) for i in x]

        self.setup()

    def setup(self):
        pass

###############################################################################

##############################################################################
#`KernelTestCase2D`
##############################################################################
class KernelTestCase2D(unittest.TestCase):
    """ Base class for testing Kernels """

    def setUp(self):
        """ Basic setup for testing the kernel.
        
        Setup:
        ------
        The kernel is assumed to be placed at the origon.
        Particle spacing dx = 0.1
        
        Expected Result:
        ----------------
        The kernel and it's gradient is integrated (summation) 
        over the domain, ensuring no boundary truncation. 
        The result should be 1 and 0 respectively.
        
        """

        #Set the limits for Summation.
        self.x = x = numpy.linspace(-1, 1, 21)
        self.y = numpy.linspace(-1, 1, 21)
        self.dx = dx = x[1] - x[0]
        self.hs = 2*dx

        self.pnt = Point()
        self.pnts = [Point(i) for i in x]

        self.xg, self.yg = numpy.meshgrid(self.x, self.y)
        self.pnt = Point()

        self.pnts = []
        #Create the other points.
        for i in range(21):
            xi = self.xg[i]
            for j in range(21):
                self.pnts.append(Point(xi[j], self.yg[i][0]))

        self.setup()

    def setup(self):
        pass

###############################################################################

        
###############################################################################
#`TestCubicSplineKernel1D`
###############################################################################
class TestCubicSplineKernel1D(KernelTestCase1D):
    """ Test function evaluation in multiple dimensions """
    
    def setup(self):
        self.kernel = kernels.CubicSplineKernel(dim=1)
    
    def test_fac(self):
        h = 0.01
        fac = 2./3/h
        kernel = self.kernel
        
        self.assertEqual(kernel.dim, 1)
        self.assertAlmostEqual(kernel.py_fac(h), fac, 6)  
        
    def test_function(self):
        """ Test for normalization of the kernel """

        print "CubicSplineKernel::py_function 1D Normalization",
        print " 10 decimal places"

        #Test using Summation formula.
        x = self.x; dx = self.dx; hs = self.hs; val = 0.0
   
        for p in self.pnts:
            val += self.kernel.py_function(self.pnt, p, hs) * dx

        self.assertAlmostEqual(val, 1, 10)

    def test_gradient1D(self):
        """ Test for the normalization of the kernel in 1D. """

        print "CubicSplineKernel::py_gradient 1D Normalization. ",
        print "10 decimal places"

        #Test using Summation formula.
        x = self.x; dx = self.dx; hs = self.hs; val = 0.0
        kernel = kernels.CubicSplineKernel(dim=1)

        pnt = Point()
        grad = Point()
        pnts = [Point(i) for i in x]
        
        for p in pnts:
            kernel.py_gradient(pnt, p, hs, grad) 
            val += grad.x * dx
            
            self.assertEqual(grad.y, 0)
            self.assertEqual(grad.z, 0)

        self.assertAlmostEqual(val, 0, 10)

    def test_moment1D(self):
        """ Test for the first moment of the kernel """

        print "CubicSplineKernel::py_function 1D Moment ",
        print " 10 decimal places"

        #Test using Summation formula.
        x = self.x; dx = self.dx; hs = self.hs; val = 0.0
        kernel = kernels.CubicSplineKernel(dim=1)

        pnt = Point()
        pnts = [Point(i) for i in x]
        
        for p in pnts:
            val += (pnt.x - p.x)*kernel.py_function(pnt, p, hs) * dx

        self.assertAlmostEqual(val, 0, 10)

    def test_grad_moment1D(self):
        """ Test for the moment of the kernel """

        print "CubicSplineKernel::py_gradient 1D Moment", 
        print "10 decimal places"
        
        x = self.x; dx = self.dx; hs = self.hs; val = 0.0
        kernel = kernels.CubicSplineKernel(dim=1)

        pnt = Point(); grad = Point()
        pnts = [Point(i) for i in x]
        
        for p in pnts:
            kernel.py_gradient(pnt, p, hs, grad) 
            val += (pnt.x - p.x)*grad.x*dx

        self.assertAlmostEqual(val, -1, 10)

    def test_(self):
        print "Testing the expression"

        x = self.x; dx = self.dx; hs = self.hs; val = 0.0
        kernel = kernels.CubicSplineKernel(dim=1)

        pnt = Point(); grad = Point()
        pnts = [Point(i) for i in x]
        
        for p in pnts:
            kernel.py_gradient(pnt, p, hs, grad)
            rab = pnt.x = p.x
            if abs(rab) >1e-16:
                val += 2*dx/(rab*rab) * (p.x - pnt.x) * rab * grad.x

        self.assertAlmostEqual(val, 0, 10)

#############################################################################

###############################################################################
#`TestCubicSplineKernel2D`
###############################################################################
class TestCubicSplineKernel2D(KernelTestCase2D):
    """ Test function evaluation in multiple dimensions """

    def setup(self):
        self.kernel = kernels.CubicSplineKernel(dim=2)

    def test_fac(self):
        dims = 1,2,3
        h = 0.01
        fac =  10/(7*numpy.pi)/(h*h)
        
        for i in range(3):
            self.assertEqual(self.kernel.dim, 2)
            self.assertAlmostEqual(self.kernel.py_fac(h), fac, 6)            
        
    def test_function2D(self):
        """ Test for the NOrmalization of the kernel in 2D """

        print "CubicSplineKernel::py_function 2D Normalization",
        print " 4 decimal places"

        #Test using Summation Formula
        xg = self.xg; yg = self.yg; dx = self.dx; hs = self.hs; val = 0.0
        kernel = self.kernel
        pnt = Point()
        pnts = []

        #Get the 2D Volume
        nvol = dx * dx
              
        for p in self.pnts:
            val += self.kernel.py_function(pnt, p, hs)*nvol

        self.assertAlmostEqual(val, 1.0, 4)

    def test_gradient2D(self):
        """ Test for the normalization of the kernel in 1D. """

        print "CubicSplineKernel::py_gradient 2D Normalization. ",
        print "10 decimal places"

        #Test using Summation Formula
        xg = self.xg; yg = self.yg; dx = self.dx; hs = self.hs
        kernel = self.kernel; pnt = self.pnt; pnts = self.pnts
        grad = Point()

        #Get the 2D Volume
        nvol = dx * dx
        val = Point()

        for p in pnts:
            kernel.py_gradient(pnt, p, hs, grad)
            val.x += grad.x*nvol
            val.y += grad.y*nvol

            self.assertEqual(grad.z, 0)

        self.assertAlmostEqual(val.x, 0, 10)
        self.assertAlmostEqual(val.y, 0, 10)

    def test_moment2D(self):
        """ Test for the first moment of the function in 2D """

        print "CubicSplineKernel::py_function 2D Moment",
        print " 10 decimal places"

        #Test using Summation Formula
        xg = self.xg; yg = self.yg; dx = self.dx; hs = self.hs
        kernel = self.kernel; pnt = self.pnt; pnts = self.pnts

        #Get the 2D Volume
        nvol = dx * dx

        val = Point()

        for p in pnts:
            val.x += (pnt.x - p.x) * (kernel.py_function(pnt, p, hs)*nvol)
            val.y += (pnt.y - p.y) * (kernel.py_function(pnt, p, hs)*nvol)

        self.assertAlmostEqual(val.x, 0, 10)
        self.assertAlmostEqual(val.y, 0, 10)


###############################################################################

###############################################################################
#`TestHarmonicKernel1D`
###############################################################################
class TestHarmonicKernel1D(KernelTestCase1D):
    """ Test function evaluation in multiple dimensions """
    
    def setup(self):
        self.kernel = kernels.HarmonicKernel(dim=1, n=1)
    
    def _test(self, func, value, places):
        x = self.x; dx = self.dx; hs = self.hs
        val = 0.0
        for p in self.pnts:
            val += func(self.pnt, p, hs) * dx
            
        self.assertAlmostEqual(val, value, places)

    def _testgrad(self, value, places):
        x = self.x; dx = self.dx; hs = self.hs
        val = Point()
        grad = Point()

        for p in self.pnts:
            self.kernel.py_gradient(self.pnt, p, hs, grad) 
            val.x += grad.x * dx
            
        self.assertAlmostEqual(val.x, value, places)
        self.assertEqual(val.y, 0)
        self.assertEqual(val.z, 0)

    def _testmom(self, value, places):
        x = self.x; dx = self.dx; hs = self.hs; pnt = self.pnt
        kernel = self.kernel
        val = Point()

        for p in self.pnts:
            val.x += (pnt.x - p.x) * kernel.py_function(pnt, p, hs)*dx
            val.y += (pnt.y - p.y) * kernel.py_function(pnt, p, hs)*dx
            
        self.assertAlmostEqual(val.x, value, places)
        self.assertEqual(val.y, value, places)
        self.assertEqual(val.z, 0)

    def test_function(self):
        """ Test for normalization of the kernel """

        print "HarmonicKernel::py_function 1D Normalization"

        #Test using Summation formula.
        kernel = self.kernel
        
        self.assertEqual(kernel.dim, 1)
        places = [1, 3, 3, 4, 5, 5, 5, 5, 5, 3]

        for i in range(1, 11):
            kernel.n = i
            self._test(kernel.py_function, 1.0, places[i-1])

    def test_gradient1D(self):
        """ Test for the normalization of the kernel in 1D. """

        print "HarmonicKernel::py_gradient 1D Normalization. "
        #Test using Summation formula.
        kernel = self.kernel
        
        self.assertEqual(kernel.dim, 1)
        places = [1, 3, 3, 4, 5, 5, 5, 5, 5, 3]

        for i in range(2, 11):
            kernel.n = i
            self._testgrad(0, places[i-1])

    def test_moment1D(self):
        """ Test for the first moment of the kernel """

        print "HarmonicKernel::py_function 1D Moment "

        #Test using Summation formula.
        kernel = self.kernel
        
        self.assertEqual(kernel.dim, 1)
        places = [1, 3, 3, 4, 5, 5, 5, 5, 5, 3]

        for i in range(1, 11):
            kernel.n = i
            self._testmom(0, places[i-1])

#############################################################################

###############################################################################
#`TestM6SplineKernel1D`
###############################################################################
class TestM6SplineKernel1D(KernelTestCase1D):
    """ Test function evaluation in multiple dimensions """
    
    def setup(self):
        self.kernel = kernels.M6SplineKernel(dim=1)
    
    def test_fac(self):
        h = 0.01
        fac = 243.0/(2560.0*h)
        kernel = self.kernel
        
        self.assertEqual(kernel.dim, 1)
        self.assertAlmostEqual(kernel.py_fac(h), fac, 6)  
        
    def test_function(self):
        """ Test for normalization of the kernel """

        print "M6SplineKernel::py_function 1D Normalization",
        print " 3 decimal places"

        #Test using Summation formula.
        x = self.x; dx = self.dx; hs = self.hs; val = 0.0
   
        for p in self.pnts:
            val += self.kernel.py_function(self.pnt, p, hs) * dx

        self.assertAlmostEqual(val, 1, 3)

    def test_gradient1D(self):
        """ Test for the normalization of the kernel in 1D. """

        print "M6SplineKernel::py_gradient 1D Normalization. ",
        print "10 decimal places"

        #Test using Summation formula.
        x = self.x; dx = self.dx; hs = self.hs; val = 0.0
        kernel = kernels.M6SplineKernel(dim=1)

        pnt = Point()
        grad = Point()
        pnts = [Point(i) for i in x]
        
        for p in pnts:
            kernel.py_gradient(pnt, p, hs, grad) 
            val += grad.x * dx
            
            self.assertEqual(grad.y, 0)
            self.assertEqual(grad.z, 0)

        self.assertAlmostEqual(val, 0, 10)

    def test_moment1D(self):
        """ Test for the first moment of the kernel """

        print "M6SplineKernel::py_function 1D Moment ",
        print " 10 decimal places"

        #Test using Summation formula.
        x = self.x; dx = self.dx; hs = self.hs; val = 0.0
        kernel = kernels.M6SplineKernel(dim=1)

        pnt = Point()
        pnts = [Point(i) for i in x]
        
        for p in pnts:
            val += (pnt.x - p.x)*kernel.py_function(pnt, p, hs) * dx

        self.assertAlmostEqual(val, 0, 10)
#############################################################################

###############################################################################
#`TestM6SplineKernel2D`
###############################################################################
class TestM6plineKernel2D(KernelTestCase2D):
    """ Test function evaluation in multiple dimensions """

    def setup(self):
        self.kernel = kernels.M6SplineKernel(dim=2)

    def test_fac(self):
        dims = 1,2,3
        h = 0.01
        fac =  (15309.0/(61184*numpy.pi*h*h))
        
        for i in range(3):
            self.assertEqual(self.kernel.dim, 2)
            self.assertAlmostEqual(self.kernel.py_fac(h), fac, 6)            
        
    def test_function2D(self):
        """ Test for the Normalization of the kernel in 2D """

        print "M6SplineKernel::py_function 2D Normalization",
        print " 4 decimal places"

        #Test using Summation Formula
        xg = self.xg; yg = self.yg; dx = self.dx; hs = self.hs; val = 0.0
        kernel = self.kernel
        pnt = Point()
        pnts = []

        #Get the 2D Volume
        nvol = dx * dx
              
        for p in self.pnts:
            val += self.kernel.py_function(pnt, p, hs)*nvol

        self.assertAlmostEqual(val, 1.0, 4)

    def test_gradient2D(self):
        """ Test for the normalization of the kernel in 1D. """

        print "M6SplineKernel::py_gradient 2D Normalization. ",
        print "10 decimal places"

        #Test using Summation Formula
        xg = self.xg; yg = self.yg; dx = self.dx; hs = self.hs
        kernel = self.kernel; pnt = self.pnt; pnts = self.pnts
        grad = Point()

        #Get the 2D Volume
        nvol = dx * dx
        val = Point()

        for p in pnts:
            kernel.py_gradient(pnt, p, hs, grad)
            val.x += grad.x*nvol
            val.y += grad.y*nvol

            self.assertEqual(grad.z, 0)

        self.assertAlmostEqual(val.x, 0, 10)
        self.assertAlmostEqual(val.y, 0, 10)

    def test_moment2D(self):
        """ Test for the first moment of the function in 2D """

        print "M6SplineKernel::py_function 2D Moment",
        print " 10 decimal places"

        #Test using Summation Formula
        xg = self.xg; yg = self.yg; dx = self.dx; hs = self.hs
        kernel = self.kernel; pnt = self.pnt; pnts = self.pnts

        #Get the 2D Volume
        nvol = dx * dx

        val = Point()

        for p in pnts:
            val.x += (pnt.x - p.x) * (kernel.py_function(pnt, p, hs)*nvol)
            val.y += (pnt.y - p.y) * (kernel.py_function(pnt, p, hs)*nvol)

        self.assertAlmostEqual(val.x, 0, 10)
        self.assertAlmostEqual(val.y, 0, 10)
###############################################################################
       
###############################################################################
#`TestGaussianKernel1D`
###############################################################################
class TestGaussianKernel1D(KernelTestCase1D):
    """ Test function evaluation in multiple dimensions """
    
    def setup(self):
        self.kernel = kernels.GaussianKernel(dim=1)
    
    def test_fac(self):
        h = 0.01
        fac = 1/((numpy.pi**.5)*h)
        kernel = self.kernel
        
        self.assertEqual(kernel.dim, 1)
        self.assertAlmostEqual(kernel.py_fac(h), fac, 6)  
        
    def test_function(self):
        """ Test for normalization of the kernel """

        print "GaussianKernel::py_function 1D Normalization",
        print " 10 decimal places"

        #Test using Summation formula.
        x = self.x; dx = self.dx; hs = self.hs; val = 0.0
   
        for p in self.pnts:
            val += self.kernel.py_function(self.pnt, p, hs) * dx

        self.assertAlmostEqual(val, 1, 10)

    def test_gradient1D(self):
        """ Test for the normalization of the kernel in 1D. """

        print "GaussianKernel::py_gradient 1D Normalization. ",
        print "10 decimal places"

        #Test using Summation formula.
        x = self.x; dx = self.dx; hs = self.hs; val = 0.0
        kernel = kernels.GaussianKernel(dim=1)

        pnt = Point()
        grad = Point()
        pnts = [Point(i) for i in x]
        
        for p in pnts:
            kernel.py_gradient(pnt, p, hs, grad) 
            val += grad.x * dx
            
            self.assertEqual(grad.y, 0)
            self.assertEqual(grad.z, 0)

        self.assertAlmostEqual(val, 0, 10)

    def test_moment1D(self):
        """ Test for the first moment of the kernel """

        print "GaussianKernel::py_function 1D Moment ",
        print " 10 decimal places"

        #Test using Summation formula.
        x = self.x; dx = self.dx; hs = self.hs; val = 0.0
        kernel = kernels.GaussianKernel(dim=1)

        pnt = Point()
        pnts = [Point(i) for i in x]
        
        for p in pnts:
            val += (pnt.x - p.x)*kernel.py_function(pnt, p, hs) * dx

        self.assertAlmostEqual(val, 0, 10)

#############################################################################

###############################################################################
#`TestGaussianKernel2D`
###############################################################################
class TestGaussianKernel2D(KernelTestCase2D):
    """ Test function evaluation in multiple dimensions """

    def setup(self):
        self.kernel = kernels.GaussianKernel(dim=2)

    def test_fac(self):
        dims = 1,2,3
        h = 0.01
        fac =1.0/(((numpy.pi**.5)*h)**2)
        
        for i in range(3):
            self.assertEqual(self.kernel.dim, 2)
            self.assertAlmostEqual(self.kernel.py_fac(h), fac, 6)            
        
    def test_function2D(self):
        """ Test for the Normalization of the kernel in 2D """

        print "GaussianKernel::py_function 2D Normalization",
        print " 4 decimal places"

        #Test using Summation Formula
        xg = self.xg; yg = self.yg; dx = self.dx; hs = self.hs; val = 0.0
        kernel = self.kernel
        pnt = Point()
        pnts = []

        #Get the 2D Volume
        nvol = dx * dx
              
        for p in self.pnts:
            val += self.kernel.py_function(pnt, p, hs)*nvol

        self.assertAlmostEqual(val, 1.0, 4)

    def test_gradient2D(self):
        """ Test for the normalization of the kernel in 1D. """

        print "GaussianKernel::py_gradient 2D Normalization. ",
        print "10 decimal places"

        #Test using Summation Formula
        xg = self.xg; yg = self.yg; dx = self.dx; hs = self.hs
        kernel = self.kernel; pnt = self.pnt; pnts = self.pnts
        grad = Point()

        #Get the 2D Volume
        nvol = dx * dx
        val = Point()

        for p in pnts:
            kernel.py_gradient(pnt, p, hs, grad)
            val.x += grad.x*nvol
            val.y += grad.y*nvol

            self.assertEqual(grad.z, 0)

        self.assertAlmostEqual(val.x, 0, 10)
        self.assertAlmostEqual(val.y, 0, 10)

    def test_moment2D(self):
        """ Test for the first moment of the function in 2D """

        print "GaussianKernel::py_function 2D Moment",
        print " 10 decimal places"

        #Test using Summation Formula
        xg = self.xg; yg = self.yg; dx = self.dx; hs = self.hs
        kernel = self.kernel; pnt = self.pnt; pnts = self.pnts

        #Get the 2D Volume
        nvol = dx * dx

        val = Point()

        for p in pnts:
            val.x += (pnt.x - p.x) * (kernel.py_function(pnt, p, hs)*nvol)
            val.y += (pnt.y - p.y) * (kernel.py_function(pnt, p, hs)*nvol)

        self.assertAlmostEqual(val.x, 0, 10)
        self.assertAlmostEqual(val.y, 0, 10)



###############################################################################

        
###############################################################################
#`W8Kernel1D`
###############################################################################
class TestW8Kernel1D(KernelTestCase1D):
    """ Test function evaluation in multiple dimensions """
    
    def setup(self):
        self.kernel = kernels.W8Kernel(dim=1)
    
    def test_fac(self):
        h = 0.01
        fac = 1.0/h
        kernel = self.kernel
        
        self.assertEqual(kernel.dim, 1)
        self.assertAlmostEqual(kernel.py_fac(h), fac, 6)  
        
    def test_function(self):
        """ Test for normalization of the kernel """

        print "W8Kernel::py_function 1D Normalization",
        print " 3 decimal places"

        #Test using Summation formula.
        x = self.x; dx = self.dx; hs = self.hs; val = 0.0
   
        for p in self.pnts:
            val += self.kernel.py_function(self.pnt, p, hs) * dx

        self.assertAlmostEqual(val, 1, 3)

    def test_gradient1D(self):
        """ Test for the normalization of the kernel in 1D. """

        print "W8Kernel::py_gradient 1D Normalization. ",
        print "4 decimal places"

        #Test using Summation formula.
        x = self.x; dx = self.dx; hs = self.hs; val = 0.0
        kernel = kernels.W8Kernel(dim=1)

        pnt = Point()
        grad = Point()
        pnts = [Point(i) for i in x]
        
        for p in pnts:
            kernel.py_gradient(pnt, p, hs, grad) 
            val += grad.x * dx
            
            self.assertEqual(grad.y, 0)
            self.assertEqual(grad.z, 0)

        self.assertAlmostEqual(val, 0, 4)

    def test_moment1D(self):
        """ Test for the first moment of the kernel """

        print "W8Kernel::py_function 1D Moment ",
        print " 5 decimal places"

        #Test using Summation formula.
        x = self.x; dx = self.dx; hs = self.hs; val = 0.0
        kernel = kernels.W8Kernel(dim=1)

        pnt = Point()
        pnts = [Point(i) for i in x]
        
        for p in pnts:
            val += (pnt.x - p.x)*kernel.py_function(pnt, p, hs) * dx

        self.assertAlmostEqual(val, 0, 5)



#############################################################################

###############################################################################
#`W8Kernel2D`
###############################################################################
class TestW8Kernel2D(KernelTestCase2D):
    """ Test function evaluation in multiple dimensions """

    def setup(self):
        self.kernel = kernels.W8Kernel(dim=2)

    def test_fac(self):
        dims = 1,2,3
        h = 0.01
        fac =  0.6348800317791645948695695182/(h*h)
        
        for i in range(3):
            self.assertEqual(self.kernel.dim, 2)
            self.assertAlmostEqual(self.kernel.py_fac(h), fac, 6)
        
    def test_function2D(self):
        """ Test for the NOrmalization of the kernel in 2D """

        print "W8Kernel::py_function 2D Normalization",
        print " 4 decimal places"

        #Test using Summation Formula
        xg = self.xg; yg = self.yg; dx = self.dx; hs = self.hs; val = 0.0
        kernel = self.kernel
        pnt = Point()
        pnts = []

        #Get the 2D Volume
        nvol = dx * dx
              
        for p in self.pnts:
            val += self.kernel.py_function(pnt, p, hs)*nvol

        self.assertAlmostEqual(val, 1.0, 4)

    def test_gradient2D(self):
        """ Test for the normalization of the kernel in 1D. """

        print "W8Kernel::py_gradient 2D Normalization. ",
        print "10 decimal places"

        #Test using Summation Formula
        xg = self.xg; yg = self.yg; dx = self.dx; hs = self.hs
        kernel = self.kernel; pnt = self.pnt; pnts = self.pnts
        grad = Point()

        #Get the 2D Volume
        nvol = dx * dx
        val = Point()

        for p in pnts:
            kernel.py_gradient(pnt, p, hs, grad)
            val.x += grad.x*nvol
            val.y += grad.y*nvol

            self.assertEqual(grad.z, 0)

        self.assertAlmostEqual(val.x, 0, 10)
        self.assertAlmostEqual(val.y, 0, 10)

    def test_moment2D(self):
        """ Test for the first moment of the function in 2D """

        print "W8Kernel::py_function 2D Moment",
        print " 5 decimal places"

        #Test using Summation Formula
        xg = self.xg; yg = self.yg; dx = self.dx; hs = self.hs
        kernel = self.kernel; pnt = self.pnt; pnts = self.pnts

        #Get the 2D Volume
        nvol = dx * dx

        val = Point()

        for p in pnts:
            val.x += (pnt.x - p.x) * (kernel.py_function(pnt, p, hs)*nvol)
            val.y += (pnt.y - p.y) * (kernel.py_function(pnt, p, hs)*nvol)

        self.assertAlmostEqual(val.x, 0, 5)
        self.assertAlmostEqual(val.y, 0, 5)

###############################################################################
###############################################################################

        
###############################################################################
#`TestW10Kernel1D`
###############################################################################

class TestW10Kernel1D(KernelTestCase1D):
    """ Test function evaluation in multiple dimensions """
    
    def setup(self):
        self.kernel = kernels.W10Kernel(dim=1)
    
    def test_fac(self):
        h = 0.01
        fac = 1.0/h
        kernel = self.kernel
        
        self.assertEqual(kernel.dim, 1)
        self.assertAlmostEqual(kernel.py_fac(h), fac, 6)  
        
    def test_function(self):
        """ Test for normalization of the kernel """

        print "W10Kernel::py_function 1D Normalization",
        print " 4 decimal places"

        #Test using Summation formula.
        x = self.x; dx = self.dx; hs = self.hs; val = 0.0
   
        for p in self.pnts:
            val += self.kernel.py_function(self.pnt, p, hs) * dx

        self.assertAlmostEqual(val, 1, 4)

    def test_gradient1D(self):
        """ Test for the normalization of the kernel in 1D. """

        print "W10Kernel::py_gradient 1D Normalization. ",
        print "4 decimal places"

        #Test using Summation formula.
        x = self.x; dx = self.dx; hs = self.hs; val = 0.0
        kernel = kernels.W10Kernel(dim=1)

        pnt = Point()
        grad = Point()
        pnts = [Point(i) for i in x]
        
        for p in pnts:
            kernel.py_gradient(pnt, p, hs, grad) 
            val += grad.x * dx
            
            self.assertEqual(grad.y, 0)
            self.assertEqual(grad.z, 0)

        self.assertAlmostEqual(val, 0, 4)

    def test_moment1D(self):
        """ Test for the first moment of the kernel """

        print "W10Kernel::py_function 1D Moment ",
        print " 4 decimal places"

        #Test using Summation formula.
        x = self.x; dx = self.dx; hs = self.hs; val = 0.0
        kernel = kernels.W10Kernel(dim=1)

        pnt = Point()
        pnts = [Point(i) for i in x]
        
        for p in pnts:
            val += (pnt.x - p.x)*kernel.py_function(pnt, p, hs) * dx

        self.assertAlmostEqual(val, 0, 4)


#############################################################################

###############################################################################
#`W10Kernel2D`
###############################################################################
class TestW10Kernel2D(KernelTestCase2D):
    """ Test function evaluation in multiple dimensions """

    def setup(self):
        self.kernel = kernels.W10Kernel(dim=2)

    def test_fac(self):
        dims = 1,2,3
        h = 0.01
        fac =  0.70546843480700509585727594599/(h*h)
        
        for i in range(3):
            self.assertEqual(self.kernel.dim, 2)
            self.assertAlmostEqual(self.kernel.py_fac(h), fac, 6)            
        
    def test_function2D(self):
        """ Test for the NOrmalization of the kernel in 2D """

        print "W10Kernel::py_function 2D Normalization",
        print " 3 decimal places"

        #Test using Summation Formula
        xg = self.xg; yg = self.yg; dx = self.dx; hs = self.hs; val = 0.0
        kernel = self.kernel
        pnt = Point()
        pnts = []

        #Get the 2D Volume
        nvol = dx * dx
        
        for p in self.pnts:
            val += self.kernel.py_function(pnt, p, hs)*nvol

        self.assertAlmostEqual(val, 1.0, 3)

    def test_gradient2D(self):
        """ Test for the normalization of the kernel in 1D. """

        print "W10Kernel::py_gradient 2D Normalization. ",
        print "5 decimal places"

        #Test using Summation Formula
        xg = self.xg; yg = self.yg; dx = self.dx; hs = self.hs
        kernel = self.kernel; pnt = self.pnt; pnts = self.pnts
        grad = Point()

        #Get the 2D Volume
        nvol = dx * dx
        val = Point()

        for p in pnts:
            kernel.py_gradient(pnt, p, hs, grad)
            val.x += grad.x*nvol
            val.y += grad.y*nvol

            self.assertEqual(grad.z, 0)

        self.assertAlmostEqual(val.x, 0, 5)
        self.assertAlmostEqual(val.y, 0, 5)

    def test_moment2D(self):
        """ Test for the first moment of the function in 2D """

        print "W10Kernel::py_function 2D Moment",
        print " 5 decimal places"

        #Test using Summation Formula
        xg = self.xg; yg = self.yg; dx = self.dx; hs = self.hs
        kernel = self.kernel; pnt = self.pnt; pnts = self.pnts

        #Get the 2D Volume
        nvol = dx * dx

        val = Point()

        for p in pnts:
            val.x += (pnt.x - p.x) * (kernel.py_function(pnt, p, hs)*nvol)
            val.y += (pnt.y - p.y) * (kernel.py_function(pnt, p, hs)*nvol)

        self.assertAlmostEqual(val.x, 0, 5)
        self.assertAlmostEqual(val.y, 0, 5)

###############################################################################

if __name__ == '__main__':
    unittest.main()
