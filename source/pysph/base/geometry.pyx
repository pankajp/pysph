""" Defines the geometry primitives used in pysph """

cimport numpy
import numpy

from pysph.base.particle_array cimport ParticleArray
from pysph.base.particles import get_particle_array
from particle_types import ParticleType

Solid = ParticleType.Solid

cdef class MeshPoint:
    """ A point object to mesh the line """

    #Declared in the .pxd file
    #cdef public Point pos
    #cdef public Point normal
    #cdef public Point tangent

    def __init__(self, Point pos, Point normal, Point tangent):
        self.pnt = pos
        self.normal = normal
        self.tangent = tangent

    def set_normal(self, Point normal):
        self.normal = normal
        
    def set_tangent(self, Point tangent):
        self.tangent = tangent

    def set_position(self, Point pos):
        self.pos = pos

    def get_distances(self, Point p):
        """ Return the normal and tangential distances to `p` """
        cdef Point pnt = self.pnt
        cdef Point pvec = p - pnt
        cdef double normal, tangent
        
        norm = pvec.dot(self.normal)
        tang = pvec.dot(self.tangent)

        return norm, tang

cdef class Line:
    """ The baisc geometric primitive used in pysph """
    
    #Declared in the .pxd file
    #cdef public Point xa
    #cdef public Point xb
    #cdef public double length
    #cdef public double angle
    #cdef public Point normal
    #cdef public Point tangent

    #The list of mesh points
    #cdef public list mesh_points

    ######################################################################
    # `object` interface.
    ######################################################################
    def __init__(self, Point xa, double length, double angle):
        """Constructor for a Line.

        Parameters:
        -----------
        origin -- The origon for the line
        length -- The length of the line
        angle -- The inclination with the positive x axis
        
        """
        self.xa = xa
        self.length = length
        self.angle = angle
        self.xb = xb = Point()
        
        cos = numpy.cos(angle)
        sin = numpy.sin(angle)
        
        #Rotate the end point of the line
        xb.x = length*cos
        xb.y = length*sin

        xb += xa
        
        #Set the tangent unit vector
        self.tangent = Point(cos, sin, 0)
        self.normal = Point(-sin, cos, 0)

    def find_closest_point(self, Point p):
        """ Given an arbitrary point `p`, find the nearest 
        point on the line.
        This may be either `xa`, `xb` or
        the perpendicular projection of p onto the line 

        """

        cdef Point pvec = p - self.xa
        cdef Point norm = self.normal
        cdef Point tang = self.tangent
        cdef double dot = pvec.dot(norm)
        cdef double length = self.length

        cdef Point pstar = pvec - (norm*dot)
        cdef double pi = pstar.dot(tang)
        
        #print self.xa, pvec, dot, pstar, pi
        if 0 <= pi <= length:
            return pstar+self.xa
        elif pi > length:
            return self.xb
        elif pi < 0:
            return self.xa

    def mesh_line(self, double dx):
        """ Mesh the line with points with the given spacing. """
        cdef double length = self.length
        cdef double x = 0.0
        cdef MeshPoint mpnt
        cdef pos, norm, tang
        cdef list mpnts = []

        norm = -self.normal
        tang = -self.tangent

        pos = self.xa + self.tangent*dx
        while (length - ((pos-self.xa).length()-1e-10)) > dx:
            mpnt = MeshPoint(pos, norm, tang)
            mpnts.append(mpnt)
            pos = pos + self.tangent*dx
        
        self.mesh_points = mpnts
        
    def _rotate_points(self):
        """ Rotate the newly meshed points """
        
        cdef double angle = self.angle
        cdef double r
        cdef MeshPoint mpnt

        for mpnt in self.mesh_points:
            r = mpnt.pnt.x
            mpnt.pnt.x = r*numpy.cos(self.angle)
            mpnt.pnt.y = r*numpy.sin(self.angle)
            
        #self.add_xa()
        #self.add_xb()
            
    def add_xa(self):
        """ Explicitly state if the first point is to be added. """
        xa = MeshPoint(self.xa, -self.normal, -self.tangent)
        self.mesh_points.insert(0, xa)

    def add_xb(self):
        """ Explicitly state if the last point is to be added. """
        
        cdef MeshPoint xb = MeshPoint(self.xb, -self.normal, -self.tangent)
        self.mesh_points.append(xb)

#############################################################################


#############################################################################
#`Geometry` class
#############################################################################
cdef class Geometry:
    """ The geometry in pysph is constructed as a list of lines """

    #Defined in the .pxd file
    #cdef public str name
    #cdef public list lines
    #cdef public list nlines
    #cdef public Point np
    #cdef public double dist
    #cdef public int nl
    #cdef public bool is_closed
    #cdef public list mpnts
    #cdef public list mpnts2    
    
    #Related to the meshing of the line
    #cdef public list mesh_pnts

    def __init__(self, str name = "", list lines = [],
                 bint is_closed = True):
        self.name = name
        self.lines = lines
        self.nlines = []
        self.np = Point()
        self.dist = 1e10
        self.nl = 0
        self.is_closed = is_closed
        self.mpnts = []
        self.mpnts2 = []

    def find_closest_point(self, Point pnt):
        """ Find the point closest to the specified point from the 
        geometry's list of lines. 

        """
        self.nlines = []
        self.nl = 0
        
        cdef list lines = self.lines
        assert len(lines) != 0, "The geometry has no lines"

        cdef Point p, diff

        for line in lines:
            p = line.find_closest_point(pnt)
            diff = pnt - p

            if abs(diff.length() - self.dist) < 1e-13:
                self.np = p; self.nlines.append(line)
                self.nl += 1

            elif diff.length() < self.dist:
                self.np = p; self.nlines = [line]
                self.dist = diff.length()
                self.nl = 1

            else:
                continue

    def get_tangent_normal(self, Point pnt):
        """ Get the tangent and normals for the given point """
        
        #cdef Line line, 1l, l2
        #cdef Point tangent, normal

        self.find_closest_point(pnt)
        
        if self.nl == 1:
            line = self.nlines[0]
            tangent = line.tangent
            normal = line.normal

        elif self.nl == 2:
            l1 = self.nlines[0]
            l2 = self.nlines[1]
            
            tangent = (l1.tangent + l2.tangent) * 0.5
            normal = (l1.normal + l2.normal) * 0.5

            tangent /= tangent.length()
            normal /= normal.length()

        else:
            print self.nl
            print "WTF!"

        self.btangent = tangent
        self.bnormal = normal

    def get_ratio(self, xi, xj):
        """ Return the ratio of normal distances for a source particle
        
        The idea is that once the appropriate calc object calls 
        get_tangent_normal for a particular point `xi`, we simply query 
        the geometry to give the ratio of the normal distances from 
        any other ponnt `xj`.

        """
        bnormal = self.bnormal
        np = self.np
              
        pi = xi - np
        pj = xj - np

        di = pi.dot(bnormal)
        dj = pj.dot(bnormal)

        return abs(dj/di)

    def _is_same_side(self, a, b):
        tmp = a * b

        if tmp < 0:
            return False
        elif tmp > 0:
            return True

    def mesh_geometry(self, double dx):
        """ Mesh the geometry by adding the mesh points on the lines """
        
        cdef list lines = self.lines
        cdef Line line
        cdef int nlines, i
        
        nlines = len(lines)
        self.mpnts = []  #Mesh points for the interior points
        self.mpnts2 = [] #Mesh points for the corner points
        
        #Add the mesh points for the interior poitns
        for line in lines:
            line.mesh_line(dx)
            self.mpnts.extend(line.mesh_points)

        #Handle the corner points
        if self.is_closed:
            for i in range(1, nlines):
                lm = lines[i-1]
                l = lines[i]
                
                pos = lm.xb
                norm = (lm.normal + l.normal) * -0.5
                tang = (lm.tangent + l.tangent) * -0.5

                norm.normalize()
                tang.normalize()

                mpnt  = MeshPoint(pos, norm, tang)
                self.mpnts2.append(mpnt)
                
            l = lines[-1]
            lp = lines[0]
            pos = l.xb

            norm = (lp.normal + l.normal) * -0.5
            tang = (lp.tangent + l.tangent) * -0.5

            norm.normalize()
            tang.normalize()
            
            mpnt  = MeshPoint(pos, norm, tang)
            self.mpnts2.append(mpnt)            
                
        else:
            for i in range(1, nlines-1):
                l = lines[i]
                lp = lines[i+1]
                lm = lines[i-1]

                pos = l.xa
                norm = (l.normal + lp.normal) * -0.5
                tang = (l.tangent + lp.tangent) * -0.5
                
                norm.normalize()
                tang.normalize()

                mpnt = MeshPoint(pos, norm, tang)
                self.mpnts2.append(mpnt)

                pos = l.xb

                norm = (l.normal + lm.normal) * -0.5
                tang = (l.tangent + lm.tangent) * -0.5

                norm.normalize()
                tang.normalize()

                mpnt = MeshPoint(pos, norm, tang)
                self.mpnts2.append(mpnt)

            #Now handle the end points in case of an open geometry
            l = lines[0]
            mpnt = MeshPoint(l.xb, -l.normal, -l.tangent)
            self.mpnts2.append(mpnt)

            l = lines[-1]
            mpnt = MeshPoint(l.xa, -l.normal, -l.tangent)
            self.mpnts2.append(mpnt)

    def get_particle_array(self, name="monaghan-edge", re_orient=False):
        """ Return a particle array with mesh point data """
        
        cdef MeshPoint mpnt
        cdef int i
        cdef Point pos, norm, tang
        cdef ParticleArray geom

        cdef list mpnts = self.mpnts + self.mpnts2
        cdef int nmpnt = len(mpnts)

        x = numpy.zeros(nmpnt, float)
        m = numpy.ones_like(x)
        y = numpy.zeros_like(x)
        z = numpy.zeros_like(x)
        h = numpy.ones_like(x)
        tx = numpy.zeros_like(x)
        ty = numpy.zeros_like(x)
        tz = numpy.zeros_like(x)
        nx = numpy.zeros_like(x)
        ny = numpy.zeros_like(x)
        nz = numpy.zeros_like(x)
        
        for i in range(nmpnt):
            mpnt = mpnts[i]
            pos = mpnt.pnt
            norm = mpnt.normal
            tang = mpnt.tangent

            x[i] = pos.x; y[i] = pos.y; z[i] = pos.z
            nx[i] = norm.x; ny[i] = norm.y; nz[i] = norm.z
            tx[i] = tang.x; ty[i] = tang.y; tz[i] = tang.z
            
        geom = get_particle_array(x=x, y=y, z=z, tx=tx, ty=ty,h=h,
                                  tz=tz, nx=nx, ny=ny, nz=nz, m=m,
                                  name=name, type=Solid)

        if re_orient:
            geom.tx *= -1
            geom.ty *= -1
            geom.tz *= -1
            
            geom.nx *= -1
            geom.ny *= -1
            geom.nz *= -1

        return geom

##############################################################################
