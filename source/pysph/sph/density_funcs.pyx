"""
SPH functions for density and related computation.
"""

# local imports
from pysph.base.point cimport Point
from pysph.base.kernelbase cimport KernelBase
from pysph.base.particle_array cimport ParticleArray

from pysph.sph.sph_func cimport SPHFunctionParticle1D, SPHFunctionParticle2D, \
    SPHFunctionParticle3D, make_coords_3d

################################################################################
# `SPHRho3D` class.
################################################################################
cdef class SPHRho3D(SPHFunctionParticle3D):
    """
    SPH function to compute density for 3d particles.

    All 3 coordinate arrays should be available.
    """
    def __init__(self, ParticleArray source, ParticleArray dest, 
                 str h='h', str mass='m', 
                 str coord_x='x', str coord_y='y', str coord_z='z',
                 str velx='u', str vely='v', str velz='z', setup_arrays=True,
                 *args, **kwargs):
        """
        Constructor.
        """
        SPHFunctionParticle3D.__init__(self, source, dest, h, mass, 
                                       coord_x, coord_y, coord_z,
                                       velx, vely, velz, setup_arrays, *args,
                                       **kwargs)        

    cdef void eval(self, int source_pid, int dest_pid, KernelBase kernel, double
                   *nr, double *dnr):
        """
        Compute the contribution of particle at source_pid on particle at
        dest_pid.
        """
        cdef double h = 0.5*(self.s_h.data[source_pid] + self.d_h.data[dest_pid])
        cdef Point src_position = Point()
        cdef Point dst_position = Point()

        make_coords_3d(self.s_x, self.s_y, self.s_z, src_position, source_pid)
        make_coords_3d(self.d_x, self.d_y, self.d_z, dst_position, dest_pid)
        
        cdef double w = kernel.function(src_position, dst_position, h)
        
        nr[0] += w*self.s_mass.data[source_pid]

    cpdef int output_fields(self) except -1:
        """
        Returns 1 - this computes a scalar quantity.
        """
        return 1


        
    
