""" Tests for the functions defined in sph/funcs.pyx """

import unittest, numpy
from pylab import load
from utils import generate_square

from pysph.base.api import ParticleArray, CubicSplineKernel
from pysph.sph.api import SPHPressureGradient, SPH, SPHCalc

############################################################################
#`TestCase1D` class
############################################################################
class TestCase1D(unittest.TestCase):
    """ A typpical 1D test case 

    Setup:
    ------
    101 particles in the interval [0,1] with various properties.

    """

    def setUp(self):
        self.x1 = x = numpy.linspace(0,1,101)

        self.dx = dx = x[1] - x[0]

        self.pa1 = pa1 = ParticleArray(x = {'data':x})
        pa1.add_prop('rho', 1.0)
        pa1.add_prop('h', 2*dx)
        pa1.add_prop('m', dx)
        pa1.add_prop('p', 1.0)
        pa1.add_prop('y', 0)
        pa1.add_prop('z', 0)

        pa1.add_props(['tmpx','tmpy','tmpz'])

        self.pa_list = [pa1]
        self.kernel = kernel = CubicSplineKernel(1)

        self.setup()

    def setup(self):
        pass
############################################################################


############################################################################
#`TestCase2D` class
############################################################################
class TestCase2D(unittest.TestCase):
    """ Base Class for tests in 2D.

    Setup:
    ------
    Nine ParticleArrays in the rectangle defined by [(-1,-1), (1, 1)].
    Tests for functions are performed on a central ParticleArray defined 
    by [(-.25, -.25), (.25, .25)]
    
    """

    def setUp(self):

        self.dx = dx = 0.05
        x1, y1 = generate_square((-.25, -.25), (.25, .25), dx, dx)
        x2, y2 = generate_square((.25, .25), (.5, .5), dx, dx)
        x3, y3 = generate_square((.25, -.25), (.5, .25), dx, dx)
        x4, y4 = generate_square((.25, -.5), (.5, -.25), dx, dx)
        x5, y5 = generate_square((-.25, -.5), (.25, -.25), dx, dx)
        x6, y6 = generate_square((-.5, -.5), (-.25, -.25), dx, dx)
        x7, y7 = generate_square((-.5, -.25), (-.25, .25), dx, dx)
        x8, y8 = generate_square((-.5, .25), (-.25, .5), dx, dx)
        x9, y9 = generate_square((-.25, .25), (.25, .5), dx, dx)

        self.pa1 = pa1 = ParticleArray(x = {'data':x1}, y = {'data':y1})
        pa1.add_prop('rho', 1.0)
        pa1.add_prop('h', 2*dx)
        pa1.add_prop('m', dx*dx)
        pa1.add_prop('p', 1.0)
        pa1.add_prop('z', 0)

        pa1.add_props(['tmpx','tmpy','tmpz'])

        self.pa2 = pa2 = ParticleArray(x = {'data':x2}, y = {'data':y2})
        pa2.add_prop('rho', 1.0)
        pa2.add_prop('h', 2*dx)
        pa2.add_prop('m', dx*dx)
        pa2.add_prop('p', 1.0)
        pa2.add_prop('z', 0)

        pa2.add_props(['tmpx','tmpy','tmpz'])

        self.pa3 = pa3 = ParticleArray(x = {'data':x3}, y = {'data':y3})
        pa3.add_prop('rho', 1.0)
        pa3.add_prop('h', 2*dx)
        pa3.add_prop('m', dx*dx)
        pa3.add_prop('p', 1.0)
        pa3.add_prop('z', 0)

        pa3.add_props(['tmpx','tmpy','tmpz'])

        self.pa4 = pa4 = ParticleArray(x = {'data':x4}, y = {'data':y4})
        pa4.add_prop('rho', 1.0)
        pa4.add_prop('h', 2*dx)
        pa4.add_prop('m', dx*dx)
        pa4.add_prop('p', 1.0)
        pa4.add_prop('z', 0)

        pa4.add_props(['tmpx','tmpy','tmpz'])

        self.pa5 = pa5 = ParticleArray(x = {'data':x5}, y = {'data':y5})
        pa5.add_prop('rho', 1.0)
        pa5.add_prop('h', 2*dx)
        pa5.add_prop('m', dx*dx)
        pa5.add_prop('p', 1.0)
        pa5.add_prop('z', 0)

        pa5.add_props(['tmpx','tmpy','tmpz'])

        self.pa6 = pa6 = ParticleArray(x = {'data':x6}, y = {'data':y6})
        pa6.add_prop('rho', 1.0)
        pa6.add_prop('h', 2*dx)
        pa6.add_prop('m', dx*dx)
        pa6.add_prop('p', 1.0)
        pa6.add_prop('z', 0)

        pa6.add_props(['tmpx','tmpy','tmpz'])

        self.pa7 = pa7 = ParticleArray(x = {'data':x7}, y = {'data':y7})
        pa7.add_prop('rho', 1.0)
        pa7.add_prop('h', 2*dx)
        pa7.add_prop('m', dx*dx)
        pa7.add_prop('p', 1.0)
        pa7.add_prop('z', 0)

        pa7.add_props(['tmpx','tmpy','tmpz'])

        self.pa8 = pa8 = ParticleArray(x = {'data':x8}, y = {'data':y8})
        pa8.add_prop('rho', 1.0)
        pa8.add_prop('h', 2*dx)
        pa8.add_prop('m', dx*dx)
        pa8.add_prop('p', 1.0)
        pa8.add_prop('z', 0)

        pa8.add_props(['tmpx','tmpy','tmpz'])

        self.pa9 = pa9 = ParticleArray(x = {'data':x9}, y = {'data':y9})
        pa9.add_prop('rho', 1.0)
        pa9.add_prop('h', 2*dx)
        pa9.add_prop('m', dx*dx)
        pa9.add_prop('p', 1.0)
        pa9.add_prop('z', 0)

        pa9.add_props(['tmpx','tmpy','tmpz'])


        self.pa_list = [pa1, pa2, pa3, pa4, pa5, pa6, pa7, pa8, pa9]
        self.kernel = kernel = CubicSplineKernel(2)

        self.setup()

    def setup(self):
        pass
############################################################################


############################################################################
#`SPH1D` class
############################################################################
class SPH1D(TestCase1D):
    """ Tests for SPH interpolation in 1D. 

    Setup:
    ------
    101 particles in the interval [0, 1] with unit density and a known
    distribution of some prop.

    Expected Output:
    ----------------
    SPH interpolation should be second order accurate for smooth datasets.

    Method of Conmarison:
    ---------------------
    "exact" solution from old 1D code is used to validate the data.

    """

    def setup(self):
        self.pa1.add_prop('f', 1.0)
        f1 = SPH(source = self.pa1, dest = self.pa1, prop_name = 'f')
        self.func_list = [f1]

        self.calc = SPHCalc(srcs = self.pa_list, dest = self.pa1, 
                            kernel = self.kernel, sph_funcs = self.func_list)

    def test_constructor(self):
        calc = self.calc
        dest = calc.dest

        np = dest.get_number_of_particles()
        for i in range(np):
            self.assertAlmostEqual(dest.f[i], 1.0, 10)

    def test_constant(self):
        exact = load('test_data/SPHConst1D.dat')

        calc = self.calc
        dst = calc.dest

        calc.sph(['tmpx','tmpy','tmpz'], False)
                
        for i, j in enumerate(dst.tmpx):
            self.assertAlmostEqual(j, exact[i], 10)

    def test_linear(self):
        exact = load('test_data/SPHLinear1D.dat')
        pa1 = self.pa1 
        pa1.set(f = pa1.x)

        calc = self.calc
        dst = calc.dest

        calc.sph(['tmpx','tmpy','tmpz'], False)
                
        for i, j in enumerate(dst.tmpx):
            self.assertAlmostEqual(j, exact[i], 10)

############################################################################



############################################################################
#`PressureGradient1D` class
############################################################################
class PressureGradient1D(TestCase1D):
    """ Test for the Symmetric pressure gradient component in 1D.

    Setup:
    ------
    101 particles in the interval [0,1] with unit density and pressure.

    Expected Output:
    ----------------
    Gradient of a constant field should be zero. Edge effects aside.

    Method of Comparison:
    ---------------------
    "exact" solution corresponding to old 1D sph code is contained in the
    file `pgrad1D.dat`. This is used to compare the results of the present
    case.    

    """
    
    def runTest(self):
        pass
    
    def setup(self):
        f1 = SPHPressureGradient(source = self.pa1, dest = self.pa1)
        self.func_list = [f1]

        self.calc = SPHCalc(srcs = self.pa_list, dest = self.pa1, 
                            kernel = self.kernel, sph_funcs = self.func_list)

        self.exact = load('test_data/pgrad1D.dat')

    def test_constructor(self):
        calc = self.calc
        
        self.assertEqual(len(calc.srcs), 1)
        self.assertEqual(calc.srcs,self.pa_list)
        self.assertEqual(calc.sph_funcs, self.func_list)
        self.assertEqual(calc.dest, self.pa1)
        self.assertEqual(calc.srcs[0], self.pa1)
        self.assertEqual(calc.kernel, self.kernel)
        self.assertEqual(calc.dim, self.kernel.dim)
        self.assertEqual(calc.h, 'h')
        self.assertEqual(len(calc.nbr_locators), 1)

        dst = calc.dest
        np = dst.get_number_of_particles()
        
        for i in range(np):
            self.assertEqual(dst.rho[i], 1.0)
            self.assertEqual(dst.h[i], 2 * self.dx)
            self.assertEqual(dst.m[i], self.dx)
            self.assertEqual(dst.p[i], 1.0)
            self.assertEqual(dst.y[i], 0.0)
    
    def test_pressure_gradient(self):
        exact = self.exact

        calc = self.calc
        dst = calc.dest

        calc.sph(['tmpx','tmpy','tmpz'], False)
                
        for i, j in enumerate(dst.tmpx):
            self.assertAlmostEqual(j, exact[i], 10)

###########################################################################



############################################################################
#`PressureGradient2D` class
############################################################################
class PressureGradient2D(TestCase2D):
    """ Test for the Symmetric pressure gradient component in 1D.

    Setup:
    ------
    One central ParticleArray on which the gradient can be estimated 
    without edge effects.

    Expected Output:
    ----------------
    Gradient of a constant field should be zero. 
    Gradient of a linear field should be constant blh blah blah

    Method of Comparison:
    ---------------------
    Use self.Assertalmostequal to adjudge the expected value based on the
    case.

    """
    
    def runTest(self):
        pass
    
    def setup(self):

        f1 = SPHPressureGradient(source = self.pa1, dest = self.pa1)
        f2 = SPHPressureGradient(source = self.pa2, dest = self.pa1)
        f3 = SPHPressureGradient(source = self.pa3, dest = self.pa1)
        f4 = SPHPressureGradient(source = self.pa4, dest = self.pa1)
        f5 = SPHPressureGradient(source = self.pa5, dest = self.pa1)
        f6 = SPHPressureGradient(source = self.pa6, dest = self.pa1)
        f7 = SPHPressureGradient(source = self.pa7, dest = self.pa1)
        f8 = SPHPressureGradient(source = self.pa8, dest = self.pa1)
        f9 = SPHPressureGradient(source = self.pa9, dest = self.pa1)
        self.func_list = [f1, f2, f3, f4, f5, f6, f7, f8, f9]

        self.calc = SPHCalc(srcs = self.pa_list, dest = self.pa1, 
                            kernel = self.kernel, sph_funcs = self.func_list)

    def test_constructor(self):
        calc = self.calc
        
        self.assertEqual(len(calc.srcs), 9)
        self.assertEqual(calc.srcs,self.pa_list)
        self.assertEqual(calc.sph_funcs, self.func_list)
        self.assertEqual(calc.dest, self.pa1)
        self.assertEqual(calc.kernel, self.kernel)
        self.assertEqual(calc.dim, self.kernel.dim)
        self.assertEqual(calc.h, 'h')
        self.assertEqual(len(calc.nbr_locators), 9)

        for i in range(9):
            self.assertEqual(calc.srcs[i], self.pa_list[i])

        dst = calc.dest
        np = dst.get_number_of_particles()
        
        for i in range(np):
            self.assertEqual(dst.rho[i], 1.0)
            self.assertEqual(dst.h[i], 2 * self.dx)
            self.assertEqual(dst.m[i], self.dx* self.dx)
            self.assertEqual(dst.p[i], 1.0)
            self.assertEqual(dst.z[i], 0.0)

    def test_pressure_gradient(self):
        calc = self.calc
        dst = calc.dest

        calc.sph(['tmpx','tmpy','tmpz'], False)
                
        px = dst.tmpx; py = dst.tmpz
        n = len(px)

        self.assertEqual(len(px), len(py))

        for i in range(n):
            pxi = dst.tmpx[i]
            pyi = dst.tmpy[i]
            pzi = dst.tmpz[i]
            
            self.assertAlmostEqual(pxi, 0, 10)
            self.assertAlmostEqual(pyi, 0, 10)
            self.assertEqual(pzi, 0)
###########################################################################

if __name__ == '__main__':
    unittest.main()
