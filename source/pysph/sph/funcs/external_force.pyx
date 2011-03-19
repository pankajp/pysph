#cython: cdivision=True
import numpy

#############################################################################
# `GravityForce` class.
#############################################################################
cdef class GravityForce(SPHFunction):
    """ Class to compute the gravity force on a particle """ 

    #Defined in the .pxd file
    #cdef double gx, gy, gz

    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True, double gx = 0.0, 
                 double gy = 0.0, double gz = 0.0, hks=False):

        SPHFunction.__init__(self, source, dest, setup_arrays)

        self.id = 'gravityforce'
        self.tag = "velocity"

        self.gx = gx
        self.gy = gy
        self.gz = gz        

    cdef void eval_single(self, size_t dest_pid,
                          KernelBase kernel, double *result):
        """ Perform the gravity force computation """

        result[0] = self.gx
        result[1] = self.gy
        result[2] = self.gz

##############################################################################


#############################################################################
# `Vector` class.
#############################################################################
cdef class VectorForce(SPHFunction):
    """ Class to compute the vector force on a particle """ 

    #Defined in the .pxd file
    #cded double fx, fy, fx

    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=True, Point force=Point(), hks=False):

        SPHFunction.__init__(self, source, dest, setup_arrays)

        self.id = 'vectorforce'
        self.force = force

    cdef void eval_single(self, size_t dest_pid,
                          KernelBase kernel, double *result):
        """ Perform the force computation """

        result[0] = self.force.data.x
        result[1] = self.force.data.y
        result[2] = self.force.data.z

################################################################################
# `MoveCircleX` class.
################################################################################
cdef class MoveCircleX(SPHFunction):
    """ Force the x coordinate of a particle to move on a circle.  """

    #Defined in the .pxd file
    def __init__(self, ParticleArray source, ParticleArray dest, 
                 *args, **kwargs):
        """ Constructor """

        SPHFunction.__init__(self, source, dest, setup_arrays = True,
                                     *args, **kwargs)

        self.id = 'circlex'
        self.tag = "position"

    cdef void eval_single(self, size_t dest_pid,
                          KernelBase kernel, double *result):
        cdef cPoint p = cPoint(self.d_x.data[dest_pid],
                           self.d_y.data[dest_pid], self.d_z.data[dest_pid])
        angle = numpy.arccos(p.x/cPoint_length(p))

        fx = -numpy.sin(angle)
        
        if p.y < 0:
            fx *= -1

        result[0] = fx
        result[1] = 0

###########################################################################

################################################################################
# `MoveCircleY` class.
################################################################################
cdef class MoveCircleY(SPHFunction):
    """ Force the y coordinate of a particle to move on a circle.  """

    #Defined in the .pxd file
    def __init__(self, ParticleArray source, ParticleArray dest, 
                 *args, **kwargs):
        """ Constructor """

        SPHFunction.__init__(self, source, dest, setup_arrays = True)

        self.id = 'circley'
        self.tag = "position"

    cdef void eval_single(self, size_t dest_pid,
                          KernelBase kernel, double *result):
        cdef cPoint p = cPoint(self.d_x.data[dest_pid],
                           self.d_y.data[dest_pid], self.d_z.data[dest_pid])
        angle = numpy.arccos(p.x/cPoint_length(p))

        fy = numpy.cos(angle)
        
        result[0] = 0
        result[1] = fy

###########################################################################
