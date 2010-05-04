import unittest
import numpy

from pysph.base.api import brute_force_nnps, ParticleArray, Point, NNPS

def sort(index, dist):
    """Sort indices and distances in order of distance for comparison."""
    pair = zip(index, dist)
    def _cmp(x, y):
        return cmp(x[1], y[1])
    pair.sort(_cmp)
    return [x[0] for x in pair], [x[1] for x in pair]

def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-10."""
    return numpy.allclose(x, y, atol=1e-10, rtol=0)

################################################################################
# Tests for the `NNPS` class.
################################################################################
class NNPSTestCase1D(unittest.TestCase):
    
    def setUp(self):
        self.x = x = numpy.array([-0.1, 0.05, -0.15, 0.1])
        self.y = y = numpy.zeros(4)
        self.z = z = numpy.zeros(4)

        self.pa = pa = ParticleArray(x = {'data':x}, y = {'data':y}, 
                                          z = {'data':z})
        self.nnps = s = NNPS(pa, x='x', y='y', z='z')
        self.bin_size = bin_size = 0.1
        s.update(bin_size)

class TestNNPS(NNPSTestCase1D):

    def test_get_none(self):
        """Query NNPS for particles such that the retuned list is empty.
        
        This should be true in the first two cases since the query
        point is farther from the particles than the radius.

        For the last case, the raduis of interaction is too small 
        for NNPS to return any nearest particles.

        """
        nps = self.nnps
        
        r, d = nps.get_nearest_particles(Point(-0.3, 0.0, 0.0), radius=0.1)
        self.assertEqual(len(r), 0)
        self.assertEqual(len(d), 0)

        r, d = nps.get_nearest_particles(Point(0.3, 0.0, 0.0), radius=0.1)
        self.assertEqual(len(r), 0)
        self.assertEqual(len(d), 0)

        r, d = nps.get_nearest_particles(Point(0.075, 0.0, 0.0), radius=0.01)
        self.assertEqual(len(r), 0)
        self.assertEqual(len(d), 0)

    def test_get_one(self):
        """ This test should return one nearest neighbor.
        
        The first case is for the left extreeme and the second for the
        right extreeme.
        """
        nps = self.nnps

        # should find one (particle on left extreme).        
        r, d = nps.get_nearest_particles(Point(-0.5, 0.0, 0.0), radius=0.36)
        self.assertEqual(len(r), 1)
        self.assertEqual(len(d), 1)
        self.assertEqual(r, [2])
        self.assertEqual(abs(d[0] - 0.35*0.35) < 1e-12, True)

        # should find one (particle on right extreme).
        r, d = nps.get_nearest_particles(Point(0.4, 0.0, 0.0), radius=0.31)
        self.assertEqual(len(r), 1)
        self.assertEqual(len(d), 1)
        self.assertEqual(r, [3])
        self.assertEqual(abs(d[0] - 0.3*0.3) < 1e-12, True)

    def test_get_two(self):
        """This test should return two nearest neighbors.        
        """
        nps = self.nnps

        r, d = nps.get_nearest_particles(Point(0.075, 0.0, 0.0), radius=0.1)
        self.assertEqual(len(r), 2)
        self.assertEqual(len(d), 2)
        self.assertEqual(r, [1, 3])
        self.assertEqual(abs(d[0] - 0.025*0.025) < 1e-12, True)
        self.assertEqual(abs(d[1] - 0.025*0.025) < 1e-12, True)


        r, d = nps.get_nearest_particles(Point(self.x[1], 0.0, 0.0), radius=0.1,
                                         exclude_index=1)
        self.assertEqual(len(r), 1)
        self.assertEqual(len(d), 1)
        self.assertEqual(r, [3])
        self.assertEqual(abs(d[0] - 0.05*0.05) < 1e-12, True)

    def test_nnps2d(self):
        """Test nnps in 2d."""
        xa = numpy.random.random(512)
        ya = numpy.random.random(512)
        za = numpy.zeros_like(xa)
        pa = ParticleArray(x = {'data':xa}, y = {'data':ya}, z = {'data':za})

        n = 16 
        for rad in (0.01, 0.05, 0.1, 0.4, 1):
            nnps = NNPS(pa)
            nnps.update(rad)
            for i in range(n):
                x, y = numpy.random.random(2)
                pt = Point(x, y, 0.0)
                idx, dist = sort(*nnps.get_nearest_particles(pt,
                                 radius=rad))
                idx1, dist1 = sort(*brute_force_nnps(pt, rad, xa, ya, za))
                self.assertEqual(idx, idx1)
                self.assertEqual(check_array(dist, dist1), True)

    def test_nnps3d(self):
        """Test nnps in 3d."""
        xa = numpy.random.random(512)
        ya = numpy.random.random(512)
        za = numpy.random.random(512)

        pa = ParticleArray(x = {'data':xa}, y = {'data':ya}, z = {'data':za})
        n = 16
        for rad in (0.01, 0.05, 0.1, 0.4, 1):
            nnps = NNPS(pa)
            nnps.update(rad)
            for i in range(n):
                pt = Point(*numpy.random.random(3))
                idx, dist = sort(*nnps.get_nearest_particles(pt,
                                radius=rad))
                idx1, dist1 = sort(*brute_force_nnps(pt, rad, xa, ya, za))
                self.assertEqual(idx, idx1)
                self.assertEqual(check_array(dist, dist1), True)


if __name__ == '__main__':
    unittest.main()
