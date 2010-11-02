""" Declarations for the geometry file """

from point cimport Point


cdef class MeshPoint:
    """ A point object to mesh the line """

    cdef public Point pnt
    cdef public Point normal
    cdef public Point tangent

cdef class Line:
    """ A line consists of two points and an outward normal """
    
    #Declared in the .pxd file
    cdef public Point xa
    cdef public Point xb
    cdef public double length
    cdef public double angle
    cdef public Point normal
    cdef public Point tangent

    #Meshing 
    cdef public list mesh_points


cdef class Geometry:
    cdef public str name
    cdef public list lines
    cdef public list nlines
    cdef public Point np
    cdef public double dist
    cdef public int nl
    cdef public Point btangent
    cdef public Point bnormal
    cdef public bint is_closed
    cdef public list mpnts
    cdef public list mpnts2
