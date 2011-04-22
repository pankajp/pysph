#cython: cdivision=True
import numpy

from pysph.solver.cl_utils import HAS_CL, get_scalar_buffer
if HAS_CL:
    import pyopencl as cl

from pysph.base.point cimport Point, cPoint, cPoint_length, cPoint_sub, \
     cPoint_distance

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

        self.cl_kernel_src_file = "external_force.cl"
        self.cl_kernel_function_name = "GravityForce"

    cdef void eval_single(self, size_t dest_pid,
                          KernelBase kernel, double *result):
        """ Perform the gravity force computation """

        result[0] = self.gx
        result[1] = self.gy
        result[2] = self.gz

    def cl_eval(self, object queue, object context, object kernel):

        tmpx = self.dest.get_cl_buffer('tmpx')
        tmpy = self.dest.get_cl_buffer('tmpy')
        tmpz = self.dest.get_cl_buffer('tmpz')

        self.cl_kernel(queue, self.global_sizes, self.local_sizes,
                       numpy.float32(self.gx), numpy.float32(self.gy),
                       numpy.float32(self.gz), tmpx, tmpy, tmpz).wait()

#############################################################################
# `VectorForce` class.
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

        self.cl_kernel_src_file = "external_force.cl"
        self.cl_kernel_function_name = "VectorForce"

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

################################################################################
# `NBodyForce` class.
################################################################################
cdef class NBodyForce(SPHFunctionParticle):
    """ Compute the force between two particles as

    ..math::

    \vec{f} = \sum_{j=1}^{N}\frac{m_j}{\norm(x_j-x_i}^3 + \eps} \vec{x_{ji}}

    """

    #Defined in the .pxd file
    def __init__(self, ParticleArray source, ParticleArray dest, eps=1e-3,
                 *args, **kwargs):
        """ Constructor """

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True,
                                     exclude_self=True)

        self.eps = eps

        self.id = 'nbody_force'
        self.tag = "velocity"

        self.cl_kernel_src_file = "external_force.cl"
        self.cl_kernel_function_name = "NBodyForce"

    cdef void eval_single(self, size_t dest_pid, KernelBase kernel,
                          double * result):
        """ Neighbors for the NBody example are by default all neighbors
        thus we return all indices of source particles.

        A particle does not contribute to itself and so if the source
        and destination particle array are the same, no acceleration
        is recorded.

        """

        cdef ParticleArray src = self.source
        cdef ParticleArray dest = self.dest

        cdef list nbrs
        cdef int nnbrs

        result[0] = 0.0
        result[1] = 0.0
        result[2] = 0.0

        nnbrs = src.get_number_of_particles()
        nbrs = range(nnbrs)

        if not (src.name == dest.name):
            for j in range(nnbrs):
                self.eval_nbr(nbrs[j], dest_pid, kernel, result)
        else:
            for j in range(nnbrs):
                source_pid = nbrs[j]
                if not (source_pid == dest_pid):
                    self.eval_nbr(source_pid, dest_pid, kernel, result)
                    
    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid,
                   KernelBase kernel, double *nr):

        cdef double mb = self.s_m.data[source_pid]

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        cdef cPoint rba = cPoint_sub( self._src, self._dst )

        cdef double invr = 1.0/(cPoint_distance(self._src,self._dst) + self.eps)
        cdef double invr3 = invr*invr*invr

        cdef double f = mb * invr3

        nr[0] += f * rba.x
        nr[1] += f * rba.y
        nr[2] += f * rba.z

    def cl_eval(self, object queue, object context, object kernel):

        # Enqueue the OpenCL kernel for execution

        self.cl_kernel(queue, self.global_sizes, self.local_sizes,
                       numpy.int32(self.source.get_number_of_particles()),
                       numpy.int32(self.dest.name==self.source.name),
                       numpy.float32(self.eps),
                       *self.args).wait()
        
###########################################################################
