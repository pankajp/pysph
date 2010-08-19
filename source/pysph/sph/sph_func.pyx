
# include for malloc
include "stdlib.pxd"

cimport numpy

# local imports
from pysph.base.particle_array cimport ParticleArray
from pysph.base.carray cimport DoubleArray
from pysph.base.point cimport Point

from pysph.base.kernels cimport KernelBase


cdef inline void make_coords_1d(DoubleArray x, Point pnt, int pid):
    """
    Convenient and fast function to get source and destination points within a
    sph function.

    Call it once for the source and once for the destination point.

    """
    pnt.x = x.data[pid]
    pnt.y = 0.0
    pnt.z = 0.0

cdef inline void make_coords_2d(DoubleArray x, DoubleArray y, Point pnt, int pid):
    """
    Convenient and fast function to get source and destination points within a
    sph function.

    Call it once for the source and once for the destination point.

    """
    pnt.x = x.data[pid]
    pnt.y = y.data[pid]
    pnt.z = 0.0

cdef inline void make_coords_3d(DoubleArray x, DoubleArray y, DoubleArray z, Point
                                pnt, int pid): 
    """
    Convenient and fast function to get source and destination points within a
    sph function.

    Call it once for the source and once for the destination point.

    """
    pnt.x = x.data[pid]
    pnt.y = y.data[pid]
    pnt.z = z.data[pid]


def py_make_coords_1d(DoubleArray x, Point pnt, int pid):
    """
    Convenience functions to tests the inline functions in the module.
    """
    make_coords_1d(x, pnt, pid)

def py_make_coords_2d(DoubleArray x, DoubleArray y, Point pnt, int pid):
    """
    Convenience functions to tests the inline functions in the module.
    """
    make_coords_2d(x, y, pnt, pid)

def py_make_coords_3d(DoubleArray x, DoubleArray y, DoubleArray z, Point
                      pnt, int pid):
    """
    Convenience functions to test the inline functions in the module.
    """
    make_coords_3d(x, y, z, pnt, pid)

###############################################################################
# `SPHFunctionParticle` class.
###############################################################################
cdef class SPHFunctionParticle:
    """
    Base class to represent an interaction between two particles from two
    possibly different particle arrays. 

    This class requires access to particle properties of possibly two different
    entities. Since there is no particle class having all properties at one
    place, names used for various properties and arrays corresponding to those
    properties are stored in this class for fast access to property values both
    at the source and destination.
    
    This class contains names, and arrays of common properties that will be
    needed for any particle-particle interaction computation. The data within
    these arrays, can be used as *array.data[pid]*, where pid in the particle
    index, "data" is the actual c-pointer to the data.

    All source arrays are prefixed with a "s_". All destination arrays are
    prefixed by a "d_". For example the mass property of the source will be in
    the s_mass array.

    **Members Variables**
     
     - dest - the destination ParticleArray
     - source - the source ParticleArray
     - mass - string hold the name used for the mass property.
     - h - string to hold the name used for the interaction radius property.
     - rho - string to hold the name used for the density property.
     - s_h - array of the source holding the h-property.
     - d_h - array of the destination holding the h-property.
     - s_mass - array of the source holding the mass-property.
     - d_mass - array of the destination holding the mass-property.
     - d_rho - array of the destination holding the density property.
     - s_rho - array of the source holding the density property.

    **Notes**
    
     - any property, say mass, should have be in an array with the same name in
       both the source and the destination.
     - all properties have a default name associated, but these can be set as
       required in the constructor.

    """
    def __init__(self, ParticleArray source, ParticleArray dest, str h='h',
                 str mass='m', str rho='rho', bint setup_arrays=True, *args,
                  **kwargs):
        """
        Constructor.

        **Parameters**
        
         - dest - the destination particle array. Properties are computed on
           particles in this array.
         - source - the source particle array. Particles from this are the
           neighbors in the sph approximation.
         - h - the name of the array in both source and dest that holds the
           interaction radius property of the particles.
         - mass - the name of the array in both source and dest that hold the mass
           property of the particles.
         - rho - the name of the array in both source and dest that hold the
           density property of the particles.
         - setup_arrays - a flag to indicate if the setup_arrays() function
           should be called. This is True by default, but will be set to False,
           when a derived class calls the base class constructor. Since each
           derived class will also implement its version of setup_arrays(),
           calling setup arrays may actually lead to access of members before
           they are actually created.

        """
        self.dest = dest
        self.source = source

        self.mass = mass
        self.rho = rho
        self.h = h        

        self.s_mass = None
        self.s_h = None
        self.s_rho = None

        self.d_mass = None
        self.d_h = None
        self.d_rho = None

        self._pnt1 = Point()
        self._pnt2 = Point()

        if setup_arrays:
            self.setup_arrays()

    cpdef setup_arrays(self):
        """
        Gets the various property arrays from the particle arrays.
        """
        if self.source is None or self.dest is None:
            return 
        
        self.s_h = self.source.get_carray(self.h)
        self.s_mass = self.source.get_carray(self.mass)
        self.s_rho = self.source.get_carray(self.rho)

        self.d_h = self.dest.get_carray(self.h)
        self.d_mass = self.dest.get_carray(self.mass)
        self.d_rho = self.dest.get_carray(self.rho)

    cdef void eval(self, int source_pid, int dest_pid, KernelBase kernel, double
               *nr, double *dnr):
        """
        Computes the contribution of particle at source_pid on particle at
        dest_pid.

        **Parameters**

         - dest_pid - the particle at which some quantity is to be computed.
         - source_pid - the neighbor whose contribution is to be computed.
         - kernel - the kernel to be used.
         - nr - memory location to store the numerator of the result. The
         - expected size of the arrays should be equal to self.output_fields().
         - dnr - memory location to store the denominator of the result. The
         - expected size of the arrays should be equal to self.output_fields().

        """
        raise NotImplementedError, 'SPHFunctionParticle::eval'

    cpdef int output_fields(self) except -1:
        """
        Returns the number of output fields, this SPHFunctionParticle will write
        to. This does not depend on the dimension of the simulation, it just
        indicates, the size of the arrays, dnr and nr that need to be passed to
        the eval function.
        """
        raise NotImplementedError, 'SPHFunctionParticle::output_fields'

    cpdef py_eval(self, int source_pid, int dest_pid, KernelBase kernel, numpy.ndarray
                  nr, numpy.ndarray dnr):
        """
        Python wrapper for the eval function, to be used in tests.
        """
        cdef double *_nr
        cdef double *_dnr
        cdef int i
        
        _nr = <double*>malloc(sizeof(double)*self.output_fields())
        _dnr = <double*>malloc(sizeof(double)*self.output_fields())

        self.eval(source_pid, dest_pid, kernel, _nr, _dnr)

        for i in range(self.output_fields()):
            nr[i]  += _nr[i]
            dnr[i] += _dnr[i]

        free(<void*>_nr)
        free(<void*>_dnr)
    

###############################################################################
# `SPHFuncParticleUser` class.
###############################################################################
cdef class SPHFuncParticleUser(SPHFunctionParticle):
    """
    User defined SPHFunctionParticle.
    """
    def __init__(self, ParticleArray dest, ParticleArray source, str h='h',
                 str mass='m', str rho='rho', bint setup_arrays=True, *args,
                  **kwargs):
        pass

    cdef void eval(self, int source_pid, int dest_pid, KernelBase kernel, double
                   *nr, double *dnr):
        """
        Compute SPH approximation using the py_eval function.
        """
        cdef size_t num_fields = self.output_fields()
        cdef numpy.ndarray _nr = numpy.zeros(num_fields)
        cdef numpy.ndarray _dnr = numpy.zeros(num_fields)
        cdef size_t i 

        self.py_eval(source_pid, dest_pid, kernel, _nr, _dnr)

        for i in range(num_fields):
            nr[i] = _nr[i]
            dnr[i] = _dnr[i]        

    cpdef py_eval(self, int source_pid, int dest_pid, KernelBase kernel, numpy.ndarray
                  nr, numpy.ndarray dnr):
        """
        Implement this function in your derived class.
        """
        raise NotImplementedError, 'SPHFuncParticleUser::py_eval'
        
###############################################################################
# `SPHFunctionPoint` class.
###############################################################################
cdef class SPHFunctionPoint:
    """
    Base class to compute the contribution of an SPH particle, on a point in
    space.

    The class is designed on similar lines to the SPHFunctionParticle class,
    except that destination point, can be any random point. Thus no dest
    particle array is necessary here. The eval interfaces in the derived classes
    also have a different signature than that of the eval interfaces of classes
    derived from SPHFunctionParticle.

    """
    def __init__(self, ParticleArray source, str h='h', str mass='m', str
                 rho='rho', bint setup_arrays=True, *args, **kwargs):
        """
        Constructor.
        """
        self.source = source
        self.mass = mass
        self.h = h
        self.rho = rho

        self.s_mass = None
        self.s_h = None
        self.s_rho = None
        
        self._pnt1 = Point()
        self._pnt2 = Point()
        
        if setup_arrays:
            self.setup_arrays()
    
    cpdef setup_arrays(self):
        """
        """
        if self.source is None:
            return

        self.s_h = self.source.get_carray(self.h)
        self.s_mass = self.source.get_carray(self.mass)
        self.s_rho = self.source.get_carray(self.rho)

    cdef void eval(self, Point pnt, int dest_pid, KernelBase kernel, double *nr,
                   double *dnr):
        """
        Computes the contribution of particle at source_pid on point pnt.

        **Parameters**

         - pnt - the point at which some quatity is to be interpolated.
         - source_pid - the neighbor whose contribution is to be computed.
         - kernel - the kernel to be used.
         - nr - memory location to store the numerator of the result.
         - dnr - memory location to store the denominator of the result.

        """
        raise NotImplementedError, 'SPHFunctionPoint::eval'

    cpdef py_eval(self, Point pnt, int dest_pid, KernelBase kernel, numpy.ndarray
                  nr, numpy.ndarray dnr):
        """
        Python wrapper for the eval function, to be used in tests.
        """
        cdef double *_nr
        cdef double *_dnr
        cdef int i
        
        _nr = <double*>malloc(sizeof(double)*self.output_fields())
        _dnr = <double*>malloc(sizeof(double)*self.output_fields())

        self.eval(pnt, dest_pid, kernel, _nr, _dnr)

        for i in range(self.output_fields()):
            nr[i] += _nr[i]
            dnr[i] += _dnr[i]

        free(<void*>_nr)
        free(<void*>_dnr)

    cpdef int output_fields(self) except -1:
        """
        Returns the number of output fields, this SPHFunctionPoint will write
        to. This does not depend on the dimension of the simulation, it just
        indicates, the size of the arrays, dnr and nr that need to be passed to
        the eval function.
        """
        raise NotImplementedError, 'SPHFunctionPoint::output_fields'

###############################################################################
# `SPHFunctionParticle1D` class.
###############################################################################
cdef class SPHFunctionParticle1D(SPHFunctionParticle):
    """
    SPHFunctionParticle for purely 1-D simulations.

    Properties which will definitely be needed for 1-D simulations included as
    member variables.

    **Members**
     
     - coord_x - holds the name of the array, which holds the x-coordinate of
        the particles. This is the only coordinate.
     - velx - holds the name of the array, which holds the x-velociy of the
        particles.
     - s_x, d_x - the source and destination arrays holding x-coordinates.
     - s_velx, d_velx - the source and destination arrays holding the
        x-velocities.

    """
    def __init__(self, ParticleArray source, ParticleArray dest, str h='h', str
                 mass='m', rho='rho', coord_x='x', velx='u', bint
                 setup_arrays=True, *args, **kwargs):
        SPHFunctionParticle.__init__(self, source, dest, h, mass, rho, False,
                                     *args, **kwargs)

        self.coord_x = coord_x
        self.velx = velx

        self.s_x = None
        self.s_velx = None

        self.d_x = None
        self.d_velx = None
        
        if setup_arrays:
            self.setup_arrays()

    cpdef setup_arrays(self):
        """
        Gets various property arrays from the particle arrays.
        """
        if self.source is None or self.dest is None:
            return 
        
        # call the parents setup_arrays now
        SPHFunctionParticle.setup_arrays(self)
        
        self.s_x = self.source.get_carray(self.coord_x)
        self.d_x = self.dest.get_carray(self.coord_x)
        
        self.s_velx = self.source.get_carray(self.velx)
        self.d_velx = self.dest.get_carray(self.velx)

###############################################################################
# `SPHFunctionPoint1D` class.
###############################################################################
cdef class SPHFunctionPoint1D(SPHFunctionPoint):
    """
    SPHFunctionPoint for purely 1-D interpolations.

    Properties which will definitely be needed for 1-D simulations included as
    member variables.

    **Members**
     
     - coord_x - holds the name of the array, which holds the x-coordinate of
        the particles. This is the only coordinate.
     - velx - holds the name of the array, which holds the x-velociy of the
        particles.
     - s_x - the  source array holding x-coordinate.
     - s_velx - the source array holding the x-velocitie.

    """
    def __init__(self, ParticleArray source, str h='h', str
                 mass='m', rho='rho', coord_x='x', velx='u', bint
                 setup_arrays=True, *args, **kwargs):
        SPHFunctionPoint.__init__(self, source, h, mass, rho, False, *args, **kwargs)

        self.coord_x = coord_x
        self.velx = velx
        self.s_x = None
        self.s_velx = None
        
        if setup_arrays:
            self.setup_arrays()

    cpdef setup_arrays(self):
        """
        Gets various property arrays from the particle arrays.
        """
        if self.source is None:
            return 
        
        # call the parents setup_arrays now
        SPHFunctionPoint.setup_arrays(self)

        self.s_x = self.source.get_carray(self.coord_x)
        self.s_velx = self.source.get_carray(self.velx)

###############################################################################
# `SPHFunctionParticle2D` class.
###############################################################################
cdef class SPHFunctionParticle2D(SPHFunctionParticle):
    """
    SPHFunctionParticle for purely 2-D simulations.

    Properties which will definitely be needed for 2-D simulations included as
    member variables.

    **Members**
     
     - coord_x - holds the name of the array, which holds the x-coordinate of
        the particles.
     - coord_y - holds the name of the array, which holds the y-coordinate of
        the particles. 
     - velx - holds the name of the array, which holds the x-velociy of the
        particles.
     - vely - holds the name of the array, which holds the y-velociy of the
        particles.
     - s_x, d_x - the source and destination arrays holding x-coordinates.
     - s_y, d_y - the source and destination arrays holding y-coordinates.
     - s_velx, d_velx - the source and destination arrays holding the
        x-velocities.
     - s_vely, d_vely - the source and destination arrays holding the
        y-velocities.

    """
    def __init__(self, ParticleArray source, ParticleArray dest, str h='h', str
                 mass='m', rho='rho', coord_x='x', coord_y='y', velx='u',
                 vely='v', bint setup_arrays=True, *args, **kwargs):
        SPHFunctionParticle.__init__(self, source, dest, h, mass, rho, False,
                                     *args, **kwargs)

        self.coord_x = coord_x
        self.coord_y = coord_y
        self.velx = velx
        self.vely = vely

        self.s_x = None
        self.s_y = None
        self.s_velx = None
        self.s_vely = None

        self.d_x = None
        self.d_y = None
        self.d_velx = None
        self.d_vely = None

        if setup_arrays:
            self.setup_arrays()

    cpdef setup_arrays(self):
        """
        Gets various property arrays from the particle arrays.
        """
        if self.source is None or self.dest is None:
            return 
        
        # call the parents setup_arrays now
        SPHFunctionParticle.setup_arrays(self)

        self.s_x = self.source.get_carray(self.coord_x)
        self.d_x = self.dest.get_carray(self.coord_x)

        self.s_y = self.source.get_carray(self.coord_y)
        self.d_y = self.dest.get_carray(self.coord_y)
        
        self.s_velx = self.source.get_carray(self.velx)
        self.d_velx = self.dest.get_carray(self.velx)

        self.s_vely = self.source.get_carray(self.vely)
        self.d_vely = self.dest.get_carray(self.vely)

###############################################################################
# `SPHFunctionPoint2D` class.
###############################################################################
cdef class SPHFunctionPoint2D(SPHFunctionPoint):
    """
    SPHFunctionPoint for interpolating from 2-D data.

    Properties which will definitely be needed for 2-D simulations included as
    member variables.

    **Members**
     
     - coord_x, coord_y - holds the name of the array, which holds the
        x-coordinate and y-coordinate respectively of the particles. 
     - velx, vely - holds the name of the array, which holds the x-velociy and
        y-velocity of the particles.
     - s_x, s_y - the  source array holding x and y coordinate.
     - s_velx, d_vely - the source array holding the x and y velocity.

    """
    def __init__(self, ParticleArray source, str h='h', str
                 mass='m', rho='rho', coord_x='x', coord_y='y', velx='u',
                 vely='v', bint setup_arrays=True, *args, **kwargs):

        SPHFunctionPoint.__init__(self, source, h, mass, rho, False, *args, **kwargs)

        self.coord_x = coord_x
        self.coord_y = coord_y

        self.velx = velx
        self.vely = vely

        self.s_x = None
        self.s_y = None
        self.s_velx = None
        self.s_vely = None

        if setup_arrays:
            self.setup_arrays()

    cpdef setup_arrays(self):
        """
        Gets various property arrays from the particle arrays.
        """
        if self.source is None:
            return 
        
        # call the parents setup_arrays now
        SPHFunctionPoint.setup_arrays(self)

        self.s_x = self.source.get_carray(self.coord_x)
        self.s_y = self.source.get_carray(self.coord_y)

        self.s_velx = self.source.get_carray(self.velx)
        self.s_vely = self.source.get_carray(self.vely)

###############################################################################
# `SPHFunctionParticle3D` class.
###############################################################################
cdef class SPHFunctionParticle3D(SPHFunctionParticle):
    """
    SPHFunctionParticle for 3-D simulations.

    Properties which will definitely be needed for 3-D simulations included as
    member variables.

    **Members**
     
     - coord_x - holds the name of the array, which holds the x-coordinate of
        the particles.
     - coord_y - holds the name of the array, which holds the y-coordinate of
        the particles. 
     - coord_z - holds the name of the array, which holds the z-coordinate of
        the particles.
     - velx - holds the name of the array, which holds the x-velocity of the
        particles.
     - vely - holds the name of the array, which holds the y-velocity of the
        particles.
     - velz - holds the name of the array, which holds the z-velocity of the
        particles. 
     - s_x, d_x - the source and destination arrays holding x-coordinates.
     - s_y, d_y - the source and destination arrays holding y-coordinates.
     - s_z, d_z - the source and destination arrays holding z-coordinates.
     - s_velx, d_velx - the source and destination arrays holding the
        x-velocities.
     - s_vely, d_vely - the source and destination arrays holding the
        y-velocities.
     - s_velz, d_velz - the source and destination arrays holding the
        z-velocities.

    """
    def __init__(self, ParticleArray source, ParticleArray dest, str h='h', str
                 mass='m', rho='rho', 
                 coord_x='x', coord_y='y', coord_z='z',
                 velx='u', vely='v', velz='w', 
                 bint setup_arrays=True, *args, **kwargs):

        SPHFunctionParticle.__init__(self, source, dest, h, mass, rho, False,
                                     *args, **kwargs)

        self.coord_x = coord_x
        self.coord_y = coord_y
        self.coord_z = coord_z

        self.velx = velx
        self.vely = vely
        self.velz = velz

        self.s_x = None
        self.s_y = None
        self.s_z = None
        self.s_velx = None
        self.s_vely = None
        self.s_velz = None

        self.d_x = None
        self.d_y = None
        self.d_z = None
        self.d_velx = None
        self.d_vely = None
        self.d_velz = None
        
        if setup_arrays:
            self.setup_arrays()

    cpdef setup_arrays(self):
        """
        Gets various property arrays from the particle arrays.
        """
        if self.source is None or self.dest is None:
            return 
        
        # call the parents setup_arrays now
        SPHFunctionParticle.setup_arrays(self)

        self.s_x = self.source.get_carray(self.coord_x)
        self.d_x = self.dest.get_carray(self.coord_x)

        self.s_y = self.source.get_carray(self.coord_y)
        self.d_y = self.dest.get_carray(self.coord_y)

        self.s_z = self.source.get_carray(self.coord_z)
        self.d_z = self.dest.get_carray(self.coord_z)
        
        self.s_velx = self.source.get_carray(self.velx)
        self.d_velx = self.dest.get_carray(self.velx)

        self.s_vely = self.source.get_carray(self.vely)
        self.d_vely = self.dest.get_carray(self.vely)

        self.s_velz = self.source.get_carray(self.velz)
        self.d_velz = self.dest.get_carray(self.velz)

###############################################################################
# `SPHFunctionPoint3D` class.
###############################################################################
cdef class SPHFunctionPoint3D(SPHFunctionPoint):
    """
    SPHFunctionPoint for interpolating from 3-D data.

    Properties which will definitely be needed for 2-D simulations included as
    member variables.

    **Members**
     
     - coord_x, coord_y, coord_z - hold the names of the arrays, which hold the
        x, y and z-coordinate respectively of the particles. 
     - velx, vely, velz - hold the name of the array, which holds the x, y and
        z-velocity of the particles.
     - s_x, s_y, s_z- the  source array holding x,y and z coordinate.
     - s_velx, s_vely, s_velz - the source array holding the x,y and z
       velocity. 

    """
    def __init__(self, ParticleArray source, 
                 str h='h', str mass='m', rho='rho', 
                 coord_x='x', coord_y='y', coord_z='z',
                 velx='u', vely='v', velz='w',
                 bint setup_arrays=True, *args, **kwargs):

        SPHFunctionPoint.__init__(self, source, h, mass, rho, False, *args,
                                  **kwargs)

        self.coord_x = coord_x
        self.coord_y = coord_y
        self.coord_z = coord_z

        self.velx = velx
        self.vely = vely
        self.velz = velz

        self.s_x = None
        self.s_y = None
        self.s_z = None
        self.s_velx = None
        self.s_vely = None
        self.s_velz = None
        
        if setup_arrays:
            self.setup_arrays()

    cpdef setup_arrays(self):
        """
        Gets various property arrays from the particle arrays.
        """
        if self.source is None:
            return 
        
        # call the parents setup_arrays now
        SPHFunctionPoint.setup_arrays(self)

        self.s_x = self.source.get_carray(self.coord_x)
        self.s_y = self.source.get_carray(self.coord_y)
        self.s_z = self.source.get_carray(self.coord_z)

        self.s_velx = self.source.get_carray(self.velx)
        self.s_vely = self.source.get_carray(self.vely)
        self.s_velz = self.source.get_carray(self.velz)
