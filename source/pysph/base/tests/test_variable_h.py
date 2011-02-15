""" Tests for searching for particles with variable smoothing lengths """

import numpy
import unittest

import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph

class OneDTestCase(unittest.TestCase):
    """ Default setup in 1D """

    def setUp(self):
        """ 
        11 particles from 0~1 with dx = 0.1, h = 0.1
        Particle 5 has h = 0.2
                
        x  x  x  x  x  O  x  x  x  x  x    
                       
        With a kernel scale factor of 2, particle 5 should have
        neighbor indices [1,2,3,4,5,6,7,8,9]        

        """        

        x = numpy.linspace(0,1,11)
        dx = x[1] - x[0]
        
        self._h = dx + 1e-10

        h = numpy.ones_like(x) * self._h
        h[5] = 2 * self._h
        
        m = numpy.ones_like(x) * dx
        rho = numpy.ones_like(x)

        pa = base.get_particle_array(type=0, name="test",
                                     x=x, h=h, m=m,rho=rho)

        self.particles = base.Particles(arrays=[pa], variable_h=True,
                                        in_parallel=False)

    def test_constructor(self):
        
        test_array = self.particles.get_named_particle_array("test")
        x,h = test_array.get('x','h')
        
        dx = x[1] - x[0]

        self.assertEqual(len(h), 11)

        for i, j in enumerate(h):
            if i == 5:
                self.assertAlmostEqual(j, 2*self._h, 10)
            else:
                self.assertAlmostEqual(j, self._h, 10)


class Test1DNeighborSearch(OneDTestCase):
    
    def test_neighbors(self):
        
        nnps_manager = self.particles.nnps_manager
        test_array = self.particles.get_named_particle_array("test")

        locator = nnps_manager.get_neighbor_particle_locator(
            source=test_array, dest=test_array, radius_scale=2.0)

        # get the neighbors
        np = test_array.get_number_of_particles()

        _nbrs = {0:[0,1,2],
                 1:[0,1,2,3,5],
                 2:[0,1,2,3,4,5],
                 3:[1,2,3,4,5],
                 4:[2,3,4,5,6],
                 5:[1,2,3,4,5,6,7,8,9],
                 6:[4,5,6,7,8],
                 7:[5,6,7,8,9],
                 8:[5,6,7,8,9,10],
                 9:[5,7,8,9,10],
                 10:[8,9,10]}


        for i in range(np):
            nbrs = locator.py_get_nearest_particles(i).get_npy_array()
            nbrs.sort()
            self.assertEqual(list(nbrs), _nbrs[i])        

if __name__ == "__main__":
    unittest.main()
