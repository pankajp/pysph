""" Tests for the geometry file """


#Author: Kunal Puri <kunalp@aero.iitb.ac.in>
#Copyright (c) 2010, Prabhu Ramachandran

import unittest, numpy
from pysph.base.api import Point, Line, Geometry, MeshPoint

class MeshPointTestCase(unittest.TestCase):
    """ Tests for the mesh point class """
    
    def setUp(self):
        self.pnt = Point(0,0)
        self.normal = Point(0,1)
        self.tangent = Point(1)

        self.mpnt = MeshPoint(self.pnt, self.normal, self.tangent)

    def test_constructor(self):
        mpnt = self.mpnt

        mpos = mpnt.pnt
        mnormal = mpnt.normal
        mtangent = mpnt.tangent

        self.assertAlmostEqual(mpos.x, 0, 10)
        self.assertAlmostEqual(mpos.y, 0, 10)

        self.assertAlmostEqual(mnormal.x, 0.0, 10)
        self.assertAlmostEqual(mnormal.y, 1.0, 10)
        self.assertAlmostEqual(mtangent.x, 1.0, 10)

    def test_get_distances(self):
        mpnt = self.mpnt

        p = Point(1,1)
        norm, tang = mpnt.get_distances(p)
        self.assertAlmostEqual(norm, 1.0, 10)
        self.assertAlmostEqual(tang, 1.0, 10)
        
        p = Point(0,1)
        norm, tang = mpnt.get_distances(p)
        self.assertAlmostEqual(norm, 1.0, 10)
        self.assertAlmostEqual(tang, 0, 10)

        p = Point(-1, 1)
        norm, tang = mpnt.get_distances(p)
        self.assertAlmostEqual(norm, 1.0, 10)
        self.assertAlmostEqual(tang, -1, 10)
        
        p = Point(-1, 0)
        norm, tang = mpnt.get_distances(p)
        self.assertAlmostEqual(norm, 0, 10)
        self.assertAlmostEqual(tang, -1, 10)

        p = Point(-1,-1)
        norm, tang = mpnt.get_distances(p)
        self.assertAlmostEqual(norm, -1, 10)
        self.assertAlmostEqual(tang, -1, 10)

        p = Point(-1, 0)
        norm, tang = mpnt.get_distances(p)
        self.assertAlmostEqual(norm, 0, 10)
        self.assertAlmostEqual(tang, -1, 10)

        p = Point(1, -1)
        norm, tang = mpnt.get_distances(p)
        self.assertAlmostEqual(norm, -1, 10)
        self.assertAlmostEqual(tang, 1, 10)
        
        p = Point(1, 0)
        norm, tang = mpnt.get_distances(p)
        self.assertAlmostEqual(norm, 0, 10)
        self.assertAlmostEqual(tang, 1, 10)


##############################################################################
#`LineTestCase`
##############################################################################
class LineTestCase(unittest.TestCase):
    """ Test for the Line class """

    def setUp(self):
        """ Simple framework to test for a line 

        Setup:
        ------
        Construct lines in different quadrants

        Expected Result:
        ----------------
        Check for the line orientation, length and tangents.

        """

        self.xa = Point(1,1)
        self.length = numpy.sqrt(2)
        
##############################################################################

class LineQuadrants(LineTestCase):
    """ Test for the line with inclination <= 90 """

    def test_zero(self):
        """ Test for zero angle of inclination """
        angle = 0.0
        line = Line(self.xa, self.length, angle)

        self.assertEqual(line.angle, 0.0)
        self.assertAlmostEqual(line.length, numpy.sqrt(2))

        xa = line.xa
        xb = line.xb

        self.assertEqual(xa.x, 1.0)
        self.assertEqual(xa.y, 1.0)

        #Test for the end point
        self.assertAlmostEqual(xb.x, 1 + numpy.sqrt(2), 10)
        self.assertAlmostEqual(xb.y, 1.0, 10)

        tangent = line.tangent
        normal = line.normal
        
        self.assertAlmostEqual(tangent.x, 1.0, 10)
        self.assertAlmostEqual(tangent.y, 0, 10)
        self.assertAlmostEqual(normal.x, 0, 10)
        self.assertAlmostEqual(normal.y, 1.0, 10)

    def test_45(self):
        angle = numpy.pi/4
        line = Line(self.xa, self.length, angle)

        xb = line.xb
        
        self.assertAlmostEqual(xb.x, 2.0, 10)
        self.assertAlmostEqual(xb.y, 2.0, 10)

        tangent = line.tangent
        normal = line.normal
        
        self.assertAlmostEqual(tangent.x, numpy.cos(numpy.pi/4), 10)
        self.assertAlmostEqual(tangent.y, numpy.sin(numpy.pi/4), 10)
        self.assertAlmostEqual(normal.x, -numpy.sin(numpy.pi/4), 10)
        self.assertAlmostEqual(normal.y, numpy.cos(numpy.pi/4) , 10)

    def test_90(self):
        angle = numpy.pi/2

        line = Line(self.xa, self.length, angle)
        xb = line.xb

        self.assertAlmostEqual(xb.x, 1.0, 10)
        self.assertAlmostEqual(xb.y, 1+numpy.sqrt(2), 10)

        tangent = line.tangent
        normal = line.normal
        
        self.assertAlmostEqual(tangent.x, 0, 10)
        self.assertAlmostEqual(tangent.y, 1.0, 10)
        self.assertAlmostEqual(normal.x, -1, 10)
        self.assertAlmostEqual(normal.y, 0 , 10)

    def test_135(self):
        angle = 3*numpy.pi/4
        line = Line(self.xa, self.length, angle)
        xb = line.xb
        
        x = self.length*numpy.cos(angle)
        y = self.length*numpy.sin(angle)
        self.assertAlmostEqual(xb.x, self.xa.x + x, 10)
        self.assertAlmostEqual(xb.y, self.xa.y + y, 10)

        tangent = line.tangent
        normal = line.normal

        self.assertAlmostEqual(tangent.x, numpy.cos(angle), 10)
        self.assertAlmostEqual(tangent.y, numpy.sin(angle), 10)
        self.assertAlmostEqual(normal.x, -numpy.sin(angle), 10)
        self.assertAlmostEqual(normal.y, numpy.cos(angle), 10)

    def test_180(self):
        angle = numpy.pi
        line = Line(self.xa, self.length, angle)
        xb = line.xb
        
        self.assertAlmostEqual(xb.x, self.xa.x -self.length, 10)
        self.assertAlmostEqual(xb.y, self.xa.y, 10)

        tangent = line.tangent
        normal = line.normal

        self.assertAlmostEqual(tangent.x, -1, 10)
        self.assertAlmostEqual(tangent.y, 0, 10)
        self.assertAlmostEqual(normal.x, 0, 10)
        self.assertAlmostEqual(normal.y, -1, 10)

    def test_270(self):
        angle = 3*numpy.pi/2
        line = Line(self.xa, self.length, angle)
        xb = line.xb
        
        self.assertAlmostEqual(xb.x, self.xa.x, 10)
        self.assertAlmostEqual(xb.y, self.xa.y - self.length, 10)

        tangent = line.tangent
        normal = line.normal

        self.assertAlmostEqual(tangent.x, 0, 10)
        self.assertAlmostEqual(tangent.y, -1, 10)
        self.assertAlmostEqual(normal.x, 1, 10)
        self.assertAlmostEqual(normal.y, 0, 10)

    def test_360(self):
        angle = 4*numpy.pi/2
        line = Line(self.xa, self.length, angle)
        xb = line.xb
        
        self.assertAlmostEqual(xb.x, self.xa.x + self.length, 10)
        self.assertAlmostEqual(xb.y, self.xa.y, 10)

        tangent = line.tangent
        normal = line.normal

        self.assertAlmostEqual(tangent.x, 1, 10)
        self.assertAlmostEqual(tangent.y, 0, 10)
        self.assertAlmostEqual(normal.x, 0, 10)
        self.assertAlmostEqual(normal.y, 1, 10)

class TestFindClossestPoint(LineTestCase):
    """ Tests for finding the closest point on a line """

    def test_zero(self):
        """ Test when the angle is 0 """
        angle = 0.0
        line = Line(self.xa, self.length, angle)

        pnt = Point(1, 2)
        p = line.find_closest_point(pnt)
        self.assertAlmostEqual(p.x, 1.0, 10)
        self.assertAlmostEqual(p.y, 1.0, 10)

        pnt = Point(2, 2)
        p = line.find_closest_point(pnt)
        self.assertAlmostEqual(p.x, 2.0, 10)
        self.assertAlmostEqual(p.y, 1.0, 10)

        pnt = Point(1.5, 2)
        p = line.find_closest_point(pnt)
        self.assertAlmostEqual(p.x, 1.5, 10)
        self.assertAlmostEqual(p.y, 1.0, 10)

        pnt = Point(2.5, 2)
        p = line.find_closest_point(pnt)
        self.assertAlmostEqual(p.x, line.xb.x, 10)
        self.assertAlmostEqual(p.y, 1.0, 10)

        pnt = Point(0.5, 2)
        p = line.find_closest_point(pnt)
        self.assertAlmostEqual(p.x, line.xa.x, 10)
        self.assertAlmostEqual(p.y, 1.0, 10)


    def test_45(self):
        """ Test for the closest point when the line is at 45 degrees """
        angle = 0.25*numpy.pi
        line = Line(self.xa, self.length, angle)

        pnt = Point(1, 2)
        p = line.find_closest_point(pnt)
        self.assertAlmostEqual(p.x, self.xa.x+0.5, 10)
        self.assertAlmostEqual(p.y, self.xa.y+0.5, 10)

        pnt = Point(0, 2)
        p = line.find_closest_point(pnt)
        self.assertAlmostEqual(p.x, self.xa.x, 10)
        self.assertAlmostEqual(p.y, self.xa.y, 10)

        pnt = Point(2, 0)
        p = line.find_closest_point(pnt)
        self.assertAlmostEqual(p.x, self.xa.x, 10)
        self.assertAlmostEqual(p.y, self.xa.y, 10)

        pnt = Point(2, 1)
        p = line.find_closest_point(pnt)
        self.assertAlmostEqual(p.x, self.xa.x+0.5, 10)
        self.assertAlmostEqual(p.y, self.xa.y+0.5, 10)

    def test_90(self):
        """ Test for the closest point when the line is 90 degrees """

        angle = 0.5*numpy.pi
        line = Line(self.xa, self.length, angle)

        pnt = Point(0, 1.5)
        p = line.find_closest_point(pnt)
        self.assertAlmostEqual(p.x, self.xa.x, 10)
        self.assertAlmostEqual(p.y, self.xa.y+0.5, 10)

        pnt = Point(0, 0)
        p = line.find_closest_point(pnt)
        self.assertAlmostEqual(p.x, self.xa.x, 10)
        self.assertAlmostEqual(p.y, self.xa.y, 10)

        pnt = Point(1.5, 1.5)
        p = line.find_closest_point(pnt)
        self.assertAlmostEqual(p.x, self.xa.x, 10)
        self.assertAlmostEqual(p.y, self.xa.y+0.5, 10)

    def test_135(self):
        """ Test for the closest point when the line is 90 degrees """

        angle = 0.75*numpy.pi
        line = Line(self.xa, self.length, angle)

        pnt = Point(0, 1)
        p = line.find_closest_point(pnt)
        self.assertAlmostEqual(p.x, self.xa.x-0.5, 10)
        self.assertAlmostEqual(p.y, self.xa.y+0.5, 10)

        pnt = Point(1, 0)
        p = line.find_closest_point(pnt)
        self.assertAlmostEqual(p.x, self.xa.x, 10)
        self.assertAlmostEqual(p.y, self.xa.y, 10)
        
        pnt = Point(1, 2)
        p = line.find_closest_point(pnt)
        self.assertAlmostEqual(p.x, self.xa.x-0.5, 10)
        self.assertAlmostEqual(p.y, self.xa.y+0.5, 10)


    def test_235(self):
        """ Test for the closest point when the line is 90 degrees """

        angle = 1.25*numpy.pi
        line = Line(self.xa, self.length, angle)

        pnt = Point(1, 0)
        p = line.find_closest_point(pnt)
        self.assertAlmostEqual(p.x, self.xa.x-0.5, 10)
        self.assertAlmostEqual(p.y, self.xa.y-0.5, 10)

        pnt = Point(0, 1)
        p = line.find_closest_point(pnt)
        self.assertAlmostEqual(p.x, self.xa.x-0.5, 10)
        self.assertAlmostEqual(p.y, self.xa.y-0.5, 10)

        pnt = Point(2, 1)
        p = line.find_closest_point(pnt)
        self.assertAlmostEqual(p.x, self.xa.x, 10)
        self.assertAlmostEqual(p.y, self.xa.y, 10)


    def test_315(self):
        """ Test for the closest point when the line is 90 degrees """

        angle = 1.75*numpy.pi
        line = Line(self.xa, self.length, angle)

        pnt = Point(2, 1)
        p = line.find_closest_point(pnt)
        self.assertAlmostEqual(p.x, self.xa.x+0.5, 10)
        self.assertAlmostEqual(p.y, self.xa.y-0.5, 10)

        pnt = Point(1, 0)
        p = line.find_closest_point(pnt)
        self.assertAlmostEqual(p.x, self.xa.x+0.5, 10)
        self.assertAlmostEqual(p.y, self.xa.y-0.5, 10)

        pnt = Point(1, 2)
        p = line.find_closest_point(pnt)
        self.assertAlmostEqual(p.x, self.xa.x, 10)
        self.assertAlmostEqual(p.y, self.xa.y, 10)

        pnt = Point(3, 0)
        p = line.find_closest_point(pnt)
        self.assertAlmostEqual(p.x, line.xb.x, 10)
        self.assertAlmostEqual(p.y, line.xb.y, 10)

###############################################################################

##############################################################################
#`LineMeshPointsTestCase`
##############################################################################
class LineMeshPointsTestCase(unittest.TestCase):
    """ Test for the meshing for the line """

    def setUp(self):
        self.xa = xa = Point(0)
        self.length = length = 1.0
        self.angle = angle = numpy.pi/4

        self.line = Line(xa, length, angle)

    def test_mesh_line(self):
        """ Mesh the line of unit length with dx = 0.2 """
        dx = 0.2
        x = 0.0
        angle = self.angle
        line = self.line
        line.mesh_line(dx)
        line.add_xa()
        line.add_xb()
        
        mpnts = line.mesh_points
        npnts = len(mpnts)
        
        self.assertEqual(npnts, 6)
        
        for pnt in mpnts:
            xp = pnt.pnt.x
            yp = pnt.pnt.y
            
            self.assertAlmostEqual(xp, x*numpy.cos(angle), 10)
            self.assertAlmostEqual(yp, x*numpy.sin(angle), 10)
            
            x += dx

##############################################################################
#`GeometryTestCase`
##############################################################################
class GeometryTestCase(unittest.TestCase):
    """ Test for the Geometry 

    Setup:
    ------
    Four lines of unit length to construct a box:
        l2 -- (0,0) <90
        l3 -- (1,0) <180
        l4 -- (1,1) <270
        l1 -- (0,1) <0

    Expected Result:
    ----------------
    8 points are tested for the tangents and normals. These are chosen
    so that analytical comparison is easy.

    """

    def runTest(self):
        pass

    def setUp(self):
        self.l1 = l1 = Line(Point(0,0), 1.0, numpy.pi/2)
        self.l2 = l2 = Line(Point(1,0), 1.0, numpy.pi)
        self.l3 = l3 = Line(Point(1,1), 1.0, 1.5*numpy.pi)
        self.l4 = l4 = Line(Point(0,1), 1.0, 0)

        self.geometry = geometry = Geometry("Box", lines = [l1, l2, l3, l4])

        self.p1 = p1 = Point(-.5, .5)
        self.p2 = p2 = Point(-.5, 1.5)
        self.p3 = p3 = Point(.5, 1.5)
        self.p4 = p4 = Point(1.5, 1.5)
        self.p5 = p5 = Point(1.5, .5)
        self.p6 = p6 = Point(1.5, -.5)
        self.p7 = p7 = Point(.5, -.5)
        self.p8 = p8 = Point(-.5, -.5)

    def test_p1(self):
        """ Test for the nearest point, tangent and normals for p1 """
        pnt = self.p1

        geometry = self.geometry
        geometry.get_tangent_normal(pnt)
        
        np = geometry.np
        nlines = geometry.nlines
        dist = geometry.dist
        btangent = geometry.btangent
        bnormal = geometry.bnormal        

        #Test for the nearest point
        self.assertAlmostEqual(np.x, 0, 10)
        self.assertAlmostEqual(np.y, 0.5, 10)

        #Test for the nearest line
        self.assertEqual(len(nlines), 1)
        self.assertEqual(nlines[0], self.l1)
        
        #Test for the distance
        self.assertAlmostEqual(dist, 0.5, 10)
        
        #Test for the tangent and normal
        self.assertAlmostEqual(btangent.x, 0, 10)
        self.assertAlmostEqual(btangent.y, 1.0, 10)
        self.assertAlmostEqual(bnormal.x, -1, 10)
        self.assertAlmostEqual(bnormal.y, 0, 10)

    def test_p2(self):
        """ Test for the nearest point, tangent and normals for p1 """
        pnt = self.p2

        geometry = self.geometry
        geometry.get_tangent_normal(pnt)
        
        np = geometry.np
        nlines = geometry.nlines
        dist = geometry.dist
        btangent = geometry.btangent
        bnormal = geometry.bnormal        

        #Test for the nearest point
        self.assertAlmostEqual(np.x, 0, 10)
        self.assertAlmostEqual(np.y, 1.0, 10)

        #Test for the nearest line
        self.assertEqual(len(nlines), 2)
        self.assertEqual(nlines[0], self.l1)
        self.assertEqual(nlines[1], self.l4)
        
        #Test for the distance
        self.assertAlmostEqual(dist, 1./numpy.sqrt(2), 10)
        
        #Test for the tangent and normal
        self.assertAlmostEqual(btangent.x, 1./numpy.sqrt(2), 10)
        self.assertAlmostEqual(btangent.y, 1./numpy.sqrt(2), 10)
        self.assertAlmostEqual(bnormal.x, -1./numpy.sqrt(2), 10)
        self.assertAlmostEqual(bnormal.y, 1./numpy.sqrt(2), 10)

    def test_p3(self):
        """ Test for the nearest point, tangent and normals for p3 """
        pnt = self.p3

        geometry = self.geometry
        geometry.get_tangent_normal(pnt)
        
        np = geometry.np
        nlines = geometry.nlines
        dist = geometry.dist
        btangent = geometry.btangent
        bnormal = geometry.bnormal        

        #Test for the nearest point
        self.assertAlmostEqual(np.x, 0.5, 10)
        self.assertAlmostEqual(np.y, 1.0, 10)

        #Test for the nearest line
        self.assertEqual(len(nlines), 1)
        self.assertEqual(nlines[0], self.l4)
        
        #Test for the distance
        self.assertAlmostEqual(dist, 0.5, 10)
        
        #Test for the tangent and normal
        self.assertAlmostEqual(btangent.x, 1.0, 10)
        self.assertAlmostEqual(btangent.y, 0, 10)
        self.assertAlmostEqual(bnormal.x, 0, 10)
        self.assertAlmostEqual(bnormal.y, 1, 10)

    def test_p4(self):
        """ Test for the nearest point, tangent and normals for p4 """
        pnt = self.p4

        geometry = self.geometry
        geometry.get_tangent_normal(pnt)
        
        np = geometry.np
        nlines = geometry.nlines
        dist = geometry.dist
        btangent = geometry.btangent
        bnormal = geometry.bnormal        

        #Test for the nearest point
        self.assertAlmostEqual(np.x, 1.0, 10)
        self.assertAlmostEqual(np.y, 1.0, 10)

        #Test for the nearest line
        self.assertEqual(len(nlines), 2)
        self.assertEqual(nlines[0], self.l3)
        self.assertEqual(nlines[1], self.l4)
        
        #Test for the distance
        self.assertAlmostEqual(dist, 1./numpy.sqrt(2), 10)
        
        #Test for the tangent and normal
        self.assertAlmostEqual(btangent.x, 1./numpy.sqrt(2), 10)
        self.assertAlmostEqual(btangent.y, -1./numpy.sqrt(2), 10)
        self.assertAlmostEqual(bnormal.x, 1./numpy.sqrt(2), 10)
        self.assertAlmostEqual(bnormal.y, 1./numpy.sqrt(2), 10)

    def test_p5(self):
        """ Test for the nearest point, tangent and normals for p5 """
        pnt = self.p5

        geometry = self.geometry
        geometry.get_tangent_normal(pnt)
        
        np = geometry.np
        nlines = geometry.nlines
        dist = geometry.dist
        btangent = geometry.btangent
        bnormal = geometry.bnormal        

        #Test for the nearest point
        self.assertAlmostEqual(np.x, 1, 10)
        self.assertAlmostEqual(np.y, 0.5, 10)

        #Test for the nearest line
        self.assertEqual(len(nlines), 1)
        self.assertEqual(nlines[0], self.l3)
        
        #Test for the distance
        self.assertAlmostEqual(dist, 0.5, 10)
        
        #Test for the tangent and normal
        self.assertAlmostEqual(btangent.x, .0, 10)
        self.assertAlmostEqual(btangent.y, -1, 10)
        self.assertAlmostEqual(bnormal.x, 1, 10)
        self.assertAlmostEqual(bnormal.y, 0, 10)

    def test_p6(self):
        """ Test for the nearest point, tangent and normals for p6 """
        pnt = self.p6

        geometry = self.geometry
        geometry.get_tangent_normal(pnt)
        
        np = geometry.np
        nlines = geometry.nlines
        dist = geometry.dist
        btangent = geometry.btangent
        bnormal = geometry.bnormal        

        #Test for the nearest point
        self.assertAlmostEqual(np.x, 1.0, 10)
        self.assertAlmostEqual(np.y, 0, 10)

        #Test for the nearest line
        self.assertEqual(len(nlines), 2)
        self.assertEqual(nlines[0], self.l2)
        self.assertEqual(nlines[1], self.l3)
        
        #Test for the distance
        self.assertAlmostEqual(dist, 1./numpy.sqrt(2), 10)
        
        #Test for the tangent and normal
        self.assertAlmostEqual(btangent.x, -1./numpy.sqrt(2), 10)
        self.assertAlmostEqual(btangent.y, -1./numpy.sqrt(2), 10)
        self.assertAlmostEqual(bnormal.x, 1./numpy.sqrt(2), 10)
        self.assertAlmostEqual(bnormal.y, -1./numpy.sqrt(2), 10)

    def test_p7(self):
        """ Test for the nearest point, tangent and normals for p5 """
        pnt = self.p7

        geometry = self.geometry
        geometry.get_tangent_normal(pnt)
        
        np = geometry.np
        nlines = geometry.nlines
        dist = geometry.dist
        btangent = geometry.btangent
        bnormal = geometry.bnormal        

        #Test for the nearest point
        self.assertAlmostEqual(np.x, 0.5, 10)
        self.assertAlmostEqual(np.y, 0, 10)

        #Test for the nearest line
        self.assertEqual(len(nlines), 1)
        self.assertEqual(nlines[0], self.l2)
        
        #Test for the distance
        self.assertAlmostEqual(dist, 0.5, 10)
        
        #Test for the tangent and normal
        self.assertAlmostEqual(btangent.x, -1, 10)
        self.assertAlmostEqual(btangent.y, 0, 10)
        self.assertAlmostEqual(bnormal.x, 0, 10)
        self.assertAlmostEqual(bnormal.y, -1, 10)

    def test_p8(self):
        """ Test for the nearest point, tangent and normals for p8 """
        pnt = self.p8

        geometry = self.geometry
        geometry.get_tangent_normal(pnt)
        
        np = geometry.np
        nlines = geometry.nlines
        dist = geometry.dist
        btangent = geometry.btangent
        bnormal = geometry.bnormal        

        #Test for the nearest point
        self.assertAlmostEqual(np.x, 0, 10)
        self.assertAlmostEqual(np.y, 0, 10)

        #Test for the nearest line
        self.assertEqual(len(nlines), 2)
        self.assertEqual(nlines[0], self.l1)
        self.assertEqual(nlines[1], self.l2)
        
        #Test for the distance
        self.assertAlmostEqual(dist, 1./numpy.sqrt(2), 10)
        
        #Test for the tangent and normal
        self.assertAlmostEqual(btangent.x, -1./numpy.sqrt(2), 10)
        self.assertAlmostEqual(btangent.y, 1./numpy.sqrt(2), 10)
        self.assertAlmostEqual(bnormal.x, -1./numpy.sqrt(2), 10)
        self.assertAlmostEqual(bnormal.y, -1./numpy.sqrt(2), 10)

##############################################################################


##############################################################################
#`GeometryTestCase`
##############################################################################
class GeometryMeshTestCase(unittest.TestCase):
    """ Test for meshing the Geometry 

    Setup:
    ------
    Four lines of unit length to construct a box:
        l1 -- (0,0) <90
        l2 -- (1,0) <180
        l4 -- (1,1) <270
        l4 -- (0,1) <00

    Expected Result:
    ----------------
    The boundary particle spacing is taken as 0.2
    No duplication of the points should arise when the geometry is closed.

    """

    def runTest(self):
        pass

    def setUp(self):
        self.l1 = l1 = Line(Point(0,0), 1.0, numpy.pi/2)
        self.l2 = l2 = Line(Point(1,0), 1.0, numpy.pi)
        self.l3 = l3 = Line(Point(1,1), 1.0, 1.5*numpy.pi)
        self.l4 = l4 = Line(Point(0,1), 1.0, 0)

        self.geometry = geometry = Geometry("Box", lines = [l1, l2, l3, l4],
                                            is_closed=True)
        self.dx = 0.2

    def test_mesh_geometry(self):
        """ Mesh and test the geoemtry """
        
        geometry = self.geometry
        geometry.mesh_geometry(self.dx)
        
        mpnts = geometry.mpnts
        mpnts2 = geometry.mpnts2
        
        npnts = len(mpnts2)
        
        #Test for the corner points
        self.assertEqual(npnts, 4)
        for i in range(npnts):
            mpnt = mpnts2[i]
            pnt = mpnt.pnt
            norm = mpnt.normal
            tang = mpnt.tangent
            
            if i == 0:
                self.assertEqual(pnt, Point(0))
                self.assertAlmostEqual(norm.x, 0.5, 10)
                self.assertAlmostEqual(norm.y, 0.5, 10)
                self.assertAlmostEqual(tang.x, 0.5, 10)
                self.assertAlmostEqual(tang.y, -0.5, 10)
                
            elif i == 1:
                self.assertEqual(pnt, Point(1))
                self.assertAlmostEqual(norm.x, -0.5, 10)
                self.assertAlmostEqual(norm.y, 0.5, 10)
                self.assertAlmostEqual(tang.x, 0.5, 10)
                self.assertAlmostEqual(tang.y, 0.5, 10)
                
            elif i == 2:
                self.assertEqual(pnt, Point(1,1))
                self.assertAlmostEqual(norm.x, -0.5, 10)
                self.assertAlmostEqual(norm.y, -0.5, 10)
                self.assertAlmostEqual(tang.x, -0.5, 10)
                self.assertAlmostEqual(tang.y, 0.5, 10)

            elif i == 3:
                self.assertEqual(pnt, Point(0,1))
                self.assertAlmostEqual(norm.x, 0.5, 10)
                self.assertAlmostEqual(norm.y, -0.5, 10)
                self.assertAlmostEqual(tang.x, -0.5, 10)
                self.assertAlmostEqual(tang.y, -0.5, 10)

        #Test for the interior points
        npnts = len(mpnts)
        
        self.assertEqual(npnts, 16)
        for i in range(4):
            pnts = mpnts[i*4:(i+1)*4]
            for pnt in pnts:
                if i == 0:
                    self.assertAlmostEqual(pnt.normal.x, 1.0, 10)
                    self.assertAlmostEqual(pnt.tangent.y, -1, 10)
                if i == 1:
                    self.assertAlmostEqual(pnt.normal.y, 1.0, 10)
                    self.assertAlmostEqual(pnt.tangent.x, 1, 10)
                if i == 2:
                    self.assertAlmostEqual(pnt.normal.x, -1.0, 10)
                    self.assertAlmostEqual(pnt.tangent.y, 1, 10)
                if i == 3:
                    self.assertAlmostEqual(pnt.normal.y, -1.0, 10)
                    self.assertAlmostEqual(pnt.tangent.x, -1, 10)
    

if __name__ == '__main__':
    unittest.main()
