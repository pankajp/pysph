"""
Tests for the sph_calc module.
"""
# standard imports
import unittest, numpy
from pylab import load

#Local imports 
from pysph.sph.api import SPHCalc, SPHRho, \
    SPHPressureGradient

from pysph.base.api import CubicSplineKernel, Point, ParticleArray

def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-16."""
    return numpy.allclose(x, y, atol=1e-16, rtol=0)

class SPHBaseTestCase1D(unittest.TestCase):
    """ Tests for sph_calc """

    def setUp(self):
        self.x1 = x = x1 = numpy.linspace(0,1,11)
        self.x2 = x2 = x1 + 1.1
        self.x3 = x3 = x1 + 2.2

        self.dx = dx = x1[1] - x1[0]
        self.h = h = numpy.ones_like(x) * 2 * dx
        self.m = m = numpy.ones_like(x) * dx

        self.rho = numpy.ones_like(x)
        self.tmp = numpy.ones_like(x)

        self.pa1 = pa1 = ParticleArray(x = {'data':x1}, m = {'data':m}, 
                                       h = {'data':h})

        pa1.add_props(['y','z','tmpx','tmpy','tmpz','u','v','w'])

        self.pa2 = pa2 = ParticleArray(x = {'data':x2}, m = {'data':m}, 
                                       h = {'data':h})

        pa2.add_props(['y','z','tmpx','tmpy','tmpz','u','v','w'])

        self.pa3 = pa3 = ParticleArray(x = {'data':x3}, m = {'data':m}, 
                                       h = {'data':h})

        pa3.add_props(['y','z','tmpx','tmpy','tmpz','u','v','w'])

        self.pa_list = [pa1, pa2, pa3]
        
        self.kernel = kernel = CubicSplineKernel(1)
        
        f1 = SPHRho(source = pa1, dest = pa2)
        f2 = SPHRho(source = pa2, dest = pa2)
        f3 = SPHRho(source = pa3, dest = pa2)

        self.func_list = [f1, f2, f3]

        self.calc = SPHCalc(srcs = self.pa_list, dest = pa2, kernel = kernel,
                            sph_funcs = self.func_list)

    def test_constructor(self):
        """ Test for construction. """
        calc = self.calc

        self.assertEqual(calc.srcs,self.pa_list)
        self.assertEqual(calc.sph_funcs, self.func_list)
        self.assertEqual(calc.dest, self.pa2)
        self.assertEqual(calc.kernel, self.kernel)
        self.assertEqual(calc.dim, self.kernel.dim)
        self.assertEqual(calc.h, 'h')

        #Setup internals and nnps's
        nnps = calc.nbr_locators

        self.assertEqual(len(nnps), 3)
        
        dst = calc.dest
        xd = dst.x
        yd = dst.y
        zd = dst.z
        hd = dst.h
        
        radius = calc.kernel.radius() + 0.01
        neighbors = {0:{0:[],1:[], 2:[]}, 1:{0:[],1:[], 2:[]}, 
                     2:{0:[],1:[], 2:[]}, 3:{0:[],1:[], 2:[]}, 
                     4:{0:[],1:[], 2:[]}, 5:{0:[],1:[], 2:[]}, 
                     6:{0:[],1:[], 2:[]}, 7:{0:[],1:[], 2:[]}, 
                     8:{0:[],1:[], 2:[]}, 9:{0:[],1:[], 2:[]}, 
                     10:{0:[],1:[], 2:[]}}
        
        for i, nps in enumerate(nnps):
            self.assertEqual(nps._h, 2 * self.dx)
            self.assertEqual(nps._pa, self.pa_list[i])

            np = dst.get_number_of_particles()

            for pidx in range(np):                
                pi = Point(xd[pidx], yd[pidx], zd[pidx])

                indx, dsts = nps.get_nearest_particles(pi, hd[pidx]*radius, 
                                                       exclude_index = -1)
                neighbors[pidx][i].extend(indx)

        exact = {
            0:{0:[7, 8, 9, 10], 1:[0, 1, 2, 3, 4], 2:[]}, 
            1:{0:[8, 9, 10], 1:[0, 1, 2, 3, 4, 5], 2:[]}, 
            2:{0:[9, 10], 1:[0, 1, 2, 3, 4, 5, 6], 2:[]},
            3:{0:[10], 1:[0, 1, 2, 3, 4, 5, 6, 7], 2:[]},
            4:{0:[], 1:[0, 1, 2, 3, 4, 5, 6, 7, 8], 2:[]},
            5:{0:[], 1:[1, 2, 3, 4, 5, 6, 7, 8, 9], 2:[]},
            6:{0:[], 1:[2, 3, 4, 5, 6, 7, 8, 9, 10], 2:[]},
            7:{0:[], 1:[3, 4, 5 ,6, 7, 8, 9, 10], 2:[0]},
            8:{0:[], 1:[4, 5, 6, 7, 8, 9, 10], 2:[0, 1]},
            9:{0:[], 1:[5, 6, 7, 8, 9, 10], 2:[0, 1, 2]},
            10:{0:[], 1:[6, 7, 8, 9, 10], 2:[0, 1, 2, 3]}
            }

        for pid in neighbors:
            nninfo = neighbors[pid]
            
            for sid in nninfo:
                nlist = nninfo[sid]
                self.assertEqual(nlist, exact[pid][sid]) 

    def test_sph1(self):
        """ Test for summation density in SPH!"""
        calc = self.calc

        calc.sph(['tmpx','tmpy','tmpz'], False)
        dst = calc.dest

        for rho in dst.tmpx:
            self.assertAlmostEqual(rho, 1.0, 10)
############################################################################

if __name__ == '__main__':
    unittest.main()
