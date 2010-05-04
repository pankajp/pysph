"""Important C-declarations for the NNPS.  This is needed so Cython can
use the definitions when these types are declared.
"""

cimport numpy
from particle_array cimport ParticleArray
from point cimport Point

###############################################################################
# `NNPS` class.
############################################################################### 
cdef class NNPS:
    """
    This class defines a nearest neighbor particle search algorithm in
    3D for a particle manager.
    """

    cdef public ParticleArray _pa
    cdef str _xn, _yn, _zn
    cdef double _xmin, _ymin, _zmin
    cdef int _ximax, _yimax, _zimax
    cdef public dict _bin
    cdef public _h

    cpdef update(self, double bin_size)
    cpdef tuple get_nearest_particles(self, Point pnt, double radius, 
                                      long exclude_index=?)
