""" Tests for the SPHComponent and function components """
import unittest, numpy
from pylab import load
from utils import generate_square

from pysph.base.api import ParticleArray, CubicSplineKernel
from pysph.sph.api import SPH

from pysph.solver.api import Fluid, EntityTypes, SPHSourceDestMode,\
    SPHComponent, SPHSummationDensityComponent, SPHPressureGradientComponent

class SPHComponentTestCase1D(unittest.TestCase):
    """ Tests for the various grouping methods """

    def setup(self):
        pass

    def setUp(self):
        """ 

        Setup:
        ------
        Generate three fluid entities in one dimension in the
        intervals [0,1], [2,3] and [4,5].

        The default Grouping mehhod is byType
 
        SPHSourceDestMode.byNone = 0
        SPHSourceDestMode.byType = 1
        SPHSourceDestMode.byAll = 2

        Expected Result:
        ---------------
        Proper setup of the calcs and other internal data used by SPHComponent

        """
        self.x1 = x1 = numpy.linspace(0,1, 101)
        self.x2 = x2 = x1 + 2
        self.x3 = x3 = x2 + 2
        y = z = numpy.zeros_like(x1)

        self.dx = dx = x1[1] - x1[0]
        self.h = h = numpy.ones_like(x1) * 2 * dx
        self.m = m = numpy.ones_like(x1) * dx
        
        self.pa1 = pa1 = ParticleArray(x = {'data':x1}, y = {'data':y},
                                       z = {'data':z}, m = {'data':m}, 
                                       h = {'data':h})

        self.pa2 = pa2 = ParticleArray(x = {'data':x2}, y = {'data':y},
                                       z = {'data':z}, m = {'data':m}, 
                                       h = {'data':h})

        self.pa3 = pa3 = ParticleArray(x = {'data':x3}, y = {'data':y},
                                       z = {'data':z}, m = {'data':m}, 
                                       h = {'data':h})

        pa1.add_props(['u','v','w'])
        pa2.add_props(['u','v','w'])
        pa3.add_props(['u','v','w'])

        self.pa_list = [pa1, pa2, pa3]
        
        self.f1 = f1 = Fluid('f1', pa1)
        self.f2 = f2 = Fluid('f2', pa2)
        self.f3 = f3 = Fluid('f3', pa3)
        
        self.entity_list = entity_list = [f1, f2, f3]
        self.kernel = kernel = CubicSplineKernel(1)

        self.component = component = SPHComponent(entity_list = entity_list,
                                                  _mode = 1, kernel = kernel)

        component.src_types = [EntityTypes.Entity_Fluid]
        component.dst_types = [EntityTypes.Entity_Fluid]

        component.sph_func = SPH

    def test_constructor(self):
        component = self.component

        self.assertEqual(component._mode, SPHSourceDestMode.byType)
        self.assertEqual(component.src_types, [EntityTypes.Entity_Fluid])
        self.assertEqual(component.dst_types, [EntityTypes.Entity_Fluid])
        self.assertEqual(len(component.entity_list), 3)

        for i in range(3):
            entity = component.entity_list[i]
            arr = entity.get_particle_array()
            
            x = arr.get('x')
            h = arr.get('h')
            m = arr.get('m')

            dx = x[1] - x[0]
            
            for i, j in enumerate(h):
                self.assertAlmostEqual(j, 2*dx, 10)
                self.assertAlmostEqual(m[i], dx, 10)

    def test_setup_entities(self):
        component = self.component
        component._setup_entities()

        elist = component.entity_list
        
        srcs = component.srcs
        dsts = component.dsts
        
        self.assertEqual(len(srcs), 3)
        self.assertEqual(len(dsts), 3)
        
        for i in range(3):
            self.assertEqual(srcs[i], elist[i])
            self.assertEqual(dsts[i], elist[i])

    def test_setup_component(self):
        component = self.component        
        component.setup_component()

        self.assertEqual(component.setup_done, True)
        self.assertEqual(len(component.calcs), 3)

        plist = self.pa_list

        for i in range(3):
            calc = component.calcs[i]
            calc_dst = calc.dest
            self.assertEqual(calc_dst, plist[i])
            
            calc_srcs = calc.srcs
            calc_funcs = calc.sph_funcs
            self.assertEqual(len(calc_srcs), 3)
            self.assertEqual(len(calc_funcs), 3)
            
            for j in range(3):
                calc_src = calc_srcs[j]
                calc_func = calc_funcs[j]
                calc_nbrs = calc.nbr_locators
                
                self.assertEqual(len(calc_nbrs), 3)
                
                self.assertEqual(calc_src, plist[j])
                self.assertEqual(calc_func.source, plist[j])
                self.assertEqual(calc_func.dest, plist[i])

                for k in range(3):
                    nps = calc_nbrs[k]
                    self.assertEqual(nps._pa, plist[k])
###########################################################################

class SummationDensityTestCase1D(SPHComponentTestCase1D):
    """ 
    Setup:
    ------
    Generate three fluid entities in one dimension in the
    intervals [0,1], [1.01,2.01] and [2.02, 3.03].
    
    The Grouping mehhod is byType
    
    Expected Result:
    ---------------
    The density for the second fluid should be unity.

    Method of Comparison:
    ---------------------
    The soulution is compared with the result from the old 1D code.
    The solution file is `SPHRho1D.dat`
    
    """

    def runTest(self):
        pass

    def setUp(self):
        self.x1 = x1 = numpy.linspace(0,1, 101)
        self.x2 = x2 = x1 + 1.01
        self.x3 = x3 = x2 + 1.01
        y = z = numpy.zeros_like(x1)

        self.dx = dx = x1[1] - x1[0]
        self.h = h = numpy.ones_like(x1) * 2 * dx
        self.m = m = numpy.ones_like(x1) * dx
        
        self.pa1 = pa1 = ParticleArray(x = {'data':x1}, y = {'data':y},
                                       z = {'data':z}, m = {'data':m}, 
                                       h = {'data':h})

        pa1.add_props(['rho','u','v','w'])

        self.pa2 = pa2 = ParticleArray(x = {'data':x2}, y = {'data':y},
                                       z = {'data':z}, m = {'data':m}, 
                                       h = {'data':h})

        pa2.add_props(['rho','u','v','w'])

        self.pa3 = pa3 = ParticleArray(x = {'data':x3}, y = {'data':y},
                                       z = {'data':z}, m = {'data':m}, 
                                       h = {'data':h})

        pa3.add_props(['rho','u','v','w'])

        self.pa_list = [pa1, pa2, pa3]
        self.f1 = f1 = Fluid('f1', pa1)
        self.f2 = f2 = Fluid('f2', pa2)
        self.f3 = f3 = Fluid('f3', pa3)
        
        self.entity_list = entity_list = [f1, f2, f3]
        self.kernel = kernel = CubicSplineKernel(1)

        self.component = SPHSummationDensityComponent(
                                              entity_list = entity_list,
                                             _mode = 1, kernel = kernel)

        component = self.component
        component.setup_component()

        self.exact = load('test_data/SPHRho1D.dat')

    def test_constructor(self):
        component = self.component

        self.assertEqual(component._mode, SPHSourceDestMode.byType)
        self.assertEqual(component.src_types, [EntityTypes.Entity_Fluid])
        self.assertEqual(component.dst_types, [EntityTypes.Entity_Fluid])
        self.assertEqual(len(component.entity_list), 3)
        self.assertEqual(component.kernel.dim, 1)

        for i in range(3):
            entity = component.entity_list[i]
            arr = entity.get_particle_array()
            
            h = arr.get('h')
            m = arr.get('m')

            dx = self.dx
            
            for i, j in enumerate(h):
                self.assertAlmostEqual(j, 2*dx, 10)
                self.assertAlmostEqual(m[i], dx, 10)

    def test_compute(self):
        component = self.component

        component._compute()
        dsts = component.dsts
        ndst = len(dsts)
        
        val = []

        for i in range(3):
            dst = dsts[i].get_particle_array()
            n = dst.get_number_of_particles()
            val.extend(dst.tmpx)
            
        val = numpy.array(val)

        for i, j in enumerate(val):
            self.assertAlmostEqual(j, self.exact[i], 10)

#############################################################################

class SummationDensityTestCase2D(unittest.TestCase):
    """ 
    Setup:
    ------
    Nine Particle Arrays in the square (-1,-1), (1,1). We consider the 
    the particle manager in the region (-.25, -.25), (.25, .25)
    
    The Grouping mehhod is byType
    
    Expected Result:
    ---------------
    The density for the second fluid should be unity.

    Method of Comparison:
    ---------------------
    The Internal fluid density should be free from edge effects.
    
    """

    def runTest(self):
        pass

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

        self.f1 = f1 = Fluid('f1', pa1)
        self.f2 = f2 = Fluid('f2', pa2)
        self.f3 = f3 = Fluid('f3', pa3)
        self.f4 = f4 = Fluid('f3', pa4)
        self.f5 = f5 = Fluid('f3', pa5)
        self.f6 = f6 = Fluid('f3', pa6)
        self.f7 = f7 = Fluid('f3', pa7)
        self.f8 = f8 = Fluid('f3', pa8)
        self.f9 = f9 = Fluid('f3', pa9)
        
        self.entity_list = entity_list = [f1, f2, f3, f4, f5, f6, f7, f8, f9]
        self.kernel = kernel = CubicSplineKernel(2)

        self.component = SPHSummationDensityComponent(
                                              entity_list = entity_list,
                                             _mode = 1, kernel = kernel)

        component = self.component
        component.setup_component()

    def test_constructor(self):
        component = self.component

        self.assertEqual(component._mode, SPHSourceDestMode.byType)
        self.assertEqual(component.src_types, [EntityTypes.Entity_Fluid])
        self.assertEqual(component.dst_types, [EntityTypes.Entity_Fluid])
        self.assertEqual(len(component.entity_list), 9)
        self.assertEqual(component.kernel.dim, 2)

        dx = self.dx
        for i in range(9):
            entity = component.entity_list[i]
            arr = entity.get_particle_array()
            
            h = arr.get('h')
            m = arr.get('m')
          
            for i, j in enumerate(h):
                self.assertAlmostEqual(j, 2*dx, 10)
                self.assertAlmostEqual(m[i], dx*dx, 10)

    def test_compute(self):
        component = self.component

        component._compute()
        dsts = component.dsts
        ndst = len(dsts)
        
        dst1 = dsts[0]
        pa1 = dst1.get_particle_array()

        for rho in pa1.tmpx:
            self.assertAlmostEqual(rho, 1.0, 4)

############################################################################

class SPHPressureGradientComponentTestCase1D(unittest.TestCase):
    """ Tests for the SPHPressureGradientComponent class in 1D 

    Setup:
    ------
    Three fluid entities in one dimension in the
    intervals [0,1], [1.01,2.01] and [2.02, 3.03].
    
    The Grouping mehhod is byType
    
    Expected Result:
    ---------------
    Pressure gradients of simple distributions.

    Method of Comparison:
    ---------------------
    1D results may be compared with exact data.

    """

    def setUp(self):
        self.kernel = CubicSplineKernel(1)

        self.x1 = x = numpy.linspace(0, 1, 101)
        self.x2 = x2 = x + 1.01
        self.x3 = x3 = x + 2.02

        self.dx = dx = 0.01
        self.pa1 = pa1 = ParticleArray(x = {'data':x})
        
        pa1.add_prop('rho',1.0)
        pa1.add_prop('h', 2*dx)
        pa1.add_prop('m', dx)
        pa1.add_prop('p', 1.0)

        pa1.add_props(['y','z','tmpx','tmpy','tmpz', 'u', 'v', 'w'])

        self.pa2 = pa2 = ParticleArray(x = {'data':x2})
        
        pa2.add_prop('rho',1.0)
        pa2.add_prop('h', 2*dx)
        pa2.add_prop('m', dx)
        pa2.add_prop('p', 1.0)

        pa2.add_props(['y','z','tmpx','tmpy','tmpz', 'u', 'v', 'w'])

        self.pa3 = pa3 = ParticleArray(x = {'data':x3})
        
        pa3.add_prop('rho',1.0)
        pa3.add_prop('h', 2*dx)
        pa3.add_prop('m', dx)
        pa3.add_prop('p', 1.0)

        pa3.add_props(['y','z','tmpx','tmpy','tmpz', 'u', 'v', 'w'])

        self.pa_list = [pa1, pa2, pa3]
        
        self.f1 = f1 = Fluid('f1', pa1)
        self.f2 = f2 = Fluid('f2', pa2)
        self.f3 = f3 = Fluid('f3', pa3)
        
        self.entity_list = entity_list = [f1, f2, f3]
        self.kernel = kernel = CubicSplineKernel(1)

        self.component = SPHPressureGradientComponent(name = 'PGC', 
                                             entity_list = entity_list,
                                             _mode = 1, kernel = kernel)
        self.component.setup_component()
    
    def test_constant(self):
        """ Test for a constant pressure distribution. """

        exact = load('test_data/SPHGradConst1D.dat')
        self.component._compute()
        self._test(exact)

    def test_linear(self):
        """ Gradient of a linear pressure distribution. """

        exact = load('test_data/SPHGradLinear1D.dat')
        component = self.component
        dsts = component.dsts
        
        for i in range(3):
            dst = dsts[i].get_particle_array()
            dst.set(p = dst.x)
           
        component._compute()
        self._test(exact)

    def test_sin(self):
        """ Gradient of numpy.sin """

        exact = load("test_data/SPHGradSin1D.dat")
        component = self.component
        dsts = component.dsts
        
        for i in range(3):
            dst = dsts[i].get_particle_array()
            dst.set(p = numpy.sin(dst.x))
        
        component._compute()
        self._test(exact)

    def _test(self, exact):
        component = self.component
        dsts = component.dsts
        
        val = []

        self.assertEqual(len(dsts), 3)
        for i in range(3):
            dst = dsts[i].get_particle_array()
            val.extend(dst.tmpx)
            
        val = numpy.array(val)
        for i, j in enumerate(val):
            self.assertAlmostEqual(-j, exact[i], 10)

############################################################################

class SPHPressureGradientComponentTestCase2D(unittest.TestCase):
    """ Tests for the SPHPressureGradientComponent class in 1D 

    Setup:
    ------
    The standar 2D setup described in the other tests.
    
    The Grouping mehhod is byType
    
    Expected Result:
    ---------------
    Pressure gradients of simple distributions.

    Method of Comparison:
    ---------------------
    Results for the interior fluid entity is checked. This should be free 
    from boundary effects.

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

        self.f1 = f1 = Fluid('f1', pa1)
        self.f2 = f2 = Fluid('f2', pa2)
        self.f3 = f3 = Fluid('f3', pa3)
        self.f4 = f4 = Fluid('f3', pa4)
        self.f5 = f5 = Fluid('f3', pa5)
        self.f6 = f6 = Fluid('f3', pa6)
        self.f7 = f7 = Fluid('f3', pa7)
        self.f8 = f8 = Fluid('f3', pa8)
        self.f9 = f9 = Fluid('f3', pa9)
        
        self.entity_list = entity_list = [f1, f2, f3, f4, f5, f6, f7, f8, f9]
        self.kernel = kernel = CubicSplineKernel(2)

        self.component = SPHPressureGradientComponent(
                                              entity_list = entity_list,
                                             _mode = 1, kernel = kernel)

        component = self.component
        component.setup_component()

    def test_constant(self):
        """ Test for a constant pressure distribution. """

        component = self.component

        dst = component.dsts[0].get_particle_array()
        np = dst.get_number_of_particles()
        component._compute()

        for i in range(np):
            px = dst.tmpx[i]
            py = dst.tmpy[i]
            pz = dst.tmpz[i]
            
            self.assertAlmostEqual(px, 0, 10)
            self.assertAlmostEqual(px, 0, 10)
            self.assertEqual(pz, 0)    

    def test_linear(self):
        """ Gradient of a linear pressure distribution. """

        component = self.component
        dsts = component.dsts
        
        for dst in dsts:
            dpa = dst.get_particle_array()
            dpa.set(p = dpa.x + dpa.y)

        component._compute()

        dst = dsts[0].get_particle_array()
        np = dst.get_number_of_particles()

        for i in range(np):
            px = dst.tmpx[i]
            py = dst.tmpy[i]
            pz = dst.tmpz[i]

            self.assertAlmostEqual(px, 1, 2)
            self.assertAlmostEqual(py, 1, 2)
            self.assertAlmostEqual(pz, 0, 10)

#############################################################################
if __name__ == '__main__':
    unittest.main()
