"""
Tests for the basic_generators module.
"""

# standard imports
import unittest


# local imports
from pysph.base.point import Point
from pysph.base.kernel3d import CubicSpline3D
from pysph.solver.basic_generators import *
from pysph.solver.utils import check_array



################################################################################
# `TestBaicFunctions` class.
################################################################################
class TestBasicFunctions(unittest.TestCase):
    """
    Tests for module level functions in the basic_generators class.
    """
    # TODO
    pass
################################################################################
# `TestLineGenerator` class.
################################################################################
class TestLineGenerator(unittest.TestCase):
    """
    Tests for the LineGenerator class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        lg = LineGenerator()
        self.assertEqual(lg.start_point, Point(0, 0, 0))
        self.assertEqual(lg.end_point, Point(0, 0, 1))
        self.assertEqual(lg.particle_spacing, 0.05)
        self.assertEqual(lg.end_points_exact, True)
        self.assertEqual(lg.tolerance, 1e-09)


        lg = LineGenerator(start_point=Point(0, 1, 0),
                           end_point=Point(1, 0, 0),
                           particle_spacing=0.1)
        self.assertEqual(lg.start_point, Point(0, 1, 0))
        self.assertEqual(lg.end_point, Point(1, 0, 0))
        self.assertEqual(lg.particle_spacing, 0.1)
        self.assertEqual(lg.end_points_exact, True)
        self.assertEqual(lg.tolerance, 1e-09)
    
    def test_get_coords(self):
        """
        Tests the get coords function.
        """
        lg = LineGenerator(particle_spacing=0.5)

        x, y, z = lg.get_coords()
        
        self.assertEqual(check_array(x, [0, 0, 0]), True)
        self.assertEqual(check_array(y, [0, 0, 0]), True)
        self.assertEqual(check_array(z, [0, 0.5, 1.0]), True)

        lg.start_point.x = 0.0
        lg.start_point.y = 1.0
        lg.start_point.z = 0.0

        x, y, z = lg.get_coords()

        self.assertEqual(check_array(x, [0, 0, 0]), True)
        self.assertEqual(check_array(y, [1., 0.5, 0]), True)
        self.assertEqual(check_array(z, [0, 0.5, 1.]), True)        

    def test_get_particles(self):
        """
        Tests the get_particles function.
        """
        lg = LineGenerator(particle_spacing=0.5,
                           kernel=CubicSpline3D())
        
        particle_array = lg.get_particles()
        self.assertEqual(particle_array.get_number_of_particles(), 3)

        self.assertEqual(check_array(particle_array.x, [0, 0, 0]), True)
        self.assertEqual(check_array(particle_array.y, [0, 0, 0]), True)
        self.assertEqual(check_array(particle_array.z, [0, 0.5, 1.0]), True)
        self.assertEqual(check_array(particle_array.h, [0.1, 0.1, 0.1]), True)
        self.assertEqual(check_array(particle_array.rho, 
                                     [1000., 1000., 1000.]),
                         True)
        self.assertEqual(check_array(particle_array.m,
                                     [1000./318.30988618379064]*3), True)

        lg.particle_spacing=0.1
        particle_array = lg.get_particles()
        m = particle_array.m[0]
        # just make sure the mass returned is less than the previous case.
        # exact computation not done currently.
        self.assertEqual(m < 1000/318.30988618379064, True)


################################################################################
# `TestRectangleGenerator` class.
################################################################################
class TestRectangleGenerator(unittest.TestCase):
    """
    Tests for the RectangleGenerator class.
    """
    def test_constructor(self):
        """
        """
        rg = RectangleGenerator()

        self.assertEqual(rg.filled, True)
        self.assertEqual(rg.start_point, Point(0, 0, 0))
        self.assertEqual(rg.end_point, Point(1, 1, 0))
        self.assertEqual(rg.particle_spacing_x1, 0.1)
        self.assertEqual(rg.particle_spacing_x2, 0.1)
        self.assertEqual(rg.end_points_exact, True)
        self.assertEqual(rg.tolerance, 1e-09)
        
        rg = RectangleGenerator(start_point=Point(-1, 0, 0),
                                end_point=Point(1, 1, 0),
                                particle_spacing_x1=0.01,
                                particle_spacing_x2=0.01)
        self.assertEqual(rg.start_point, Point(-1, 0, 0))
        self.assertEqual(rg.end_point, Point(1,1,0))
        self.assertEqual(rg.particle_spacing_x1, 0.01)
        self.assertEqual(rg.particle_spacing_x2, 0.01)

    def test_get_coords(self):
        """
        Tests the get_coords function.
        """
        rg = RectangleGenerator(particle_spacing_x1=0.5,
                                particle_spacing_x2=0.5)

        x, y, z = rg.get_coords()
        self.assertEqual(check_array(x, [0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0,
                                         1.0 ]), True)
        self.assertEqual(check_array(y, [0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5,
                                         1.0 ]), True)
        self.assertEqual(check_array(z, [0, 0, 0, 0, 0, 0, 0, 0, 0]), True)

        rg.start_point.x = 0.0
        rg.start_point.y = 0.0
        rg.start_point.z = 0.0
        rg.end_point.x = 0.0
        rg.end_point.y = 1.0
        rg.end_point.z = 1.0

        x, y, z = rg.get_coords()

        self.assertEqual(check_array(y, [0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0,
                                         1.0 ]), True)
        self.assertEqual(check_array(z, [0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5,
                                         1.0 ]), True)
        self.assertEqual(check_array(x, [0, 0, 0, 0, 0, 0, 0, 0, 0]), True)

        rg.start_point.x = 0.0
        rg.start_point.y = 0.0
        rg.start_point.z = 0.0
        rg.end_point.x = 1.0
        rg.end_point.y = 0.0
        rg.end_point.z = 1.0

        x, y, z = rg.get_coords()

        self.assertEqual(check_array(x, [0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0,
                                         1.0 ]), True)
        self.assertEqual(check_array(z, [0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5,
                                         1.0 ]), True)
        self.assertEqual(check_array(y, [0, 0, 0, 0, 0, 0, 0, 0, 0]), True)
        
    def test_get_paricles(self):
        """
        Tests the get_partilces function.
        """
        rg = RectangleGenerator(particle_spacing_x1=0.5,
                                particle_spacing_x2=0.5,
                                kernel=CubicSpline3D())
        p = rg.get_particles()
        self.assertEqual(p.get_number_of_particles(), 9)
        self.assertEqual(check_array(p.x, [0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0,
                                         1.0 ]), True)
        self.assertEqual(check_array(p.y, [0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5,
                                         1.0 ]), True)
        self.assertEqual(check_array(p.z, [0, 0, 0, 0, 0, 0, 0, 0, 0]), True)
        self.assertEqual(check_array(p.h, [0.1]*9), True)
        self.assertEqual(check_array(p.rho, [1000.]*9), True)
        self.assertEqual(check_array(p.m, [1000./318.30988618379064]*9), True)

    def test_get_coords_empty_rectangle(self):
        """
        Tests the rectangle generator when an empty rectangle is requested.
        """
        rg = RectangleGenerator(particle_spacing_x1=0.5,
                                particle_spacing_x2=0.5,
                                filled=False)

        x, y, z = rg.get_coords()
        
        self.assertEqual(len(x), 8)
        self.assertEqual(len(y), 8)
        self.assertEqual(len(z), 8)

        self.assertEqual(check_array(x, [0, 0.5, 1.0, 0, 0, 0.5, 1.0, 1.0]),
                         True)
        self.assertEqual(check_array(y, [0, 0, 0, 0.5, 1.0, 1.0, 1.0, 0.5]),
                         True)
        self.assertEqual(check_array(z, [0, 0, 0, 0, 0, 0, 0, 0]), True)
        

    def test_get_particles_empty_rectangle(self):
        """
        Tests the get_particles for empty rectangles.
        """
        rg = RectangleGenerator(particle_spacing_x1=0.5,
                                particle_spacing_x2=0.5,
                                filled=False,
                                density_computation_mode=DCM.Set_Constant,
                                mass_computation_mode=MCM.Set_Constant,
                                particle_mass=1.0)

        p = rg.get_particles()

        self.assertEqual(check_array(p.x, [0, 0.5, 1.0, 0, 0, 0.5, 1.0, 1.0]),
                         True)
        self.assertEqual(check_array(p.y, [0, 0, 0, 0.5, 1.0, 1.0, 1.0, 0.5]),
                         True)
        self.assertEqual(check_array(p.z, [0]*8), True)
        self.assertEqual(check_array(p.h, [0.1]*8), True)
        self.assertEqual(check_array(p.rho, [1000.]*8), True)
        self.assertEqual(check_array(p.m, [1.0]*8), True)

if __name__ == '__main__':
    unittest.main()
