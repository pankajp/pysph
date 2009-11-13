"""
Module to hold some basic SPH functions.
"""

from pysph.base.carray cimport DoubleArray
from pysph.sph.sph_func cimport *


cdef class SPH3D(SPHFunctionParticle3D):
    """
    Simple SPH interpolation.
    """
    def __init__(self, ParticleArray source, ParticleArray dest,
                 str h='h', str mass='m', str rho='rho',
                 str coord_x='x', str coord_y='y', str coord_z='z',
                 str velx='u', str vely='v', str velz='z', bint
                 setup_arrays=True,
                 str prop_name='', 
                 *args, **kwargs):
        """
        Constructor.
        """
        SPHFunctionParticle3D.__init__(self, source, dest, h, mass, rho,
                                       coord_x, coord_y, coord_z,
                                       velx, vely, velz, setup_arrays=False)

        self.prop_name = prop_name

        if setup_arrays == True:
            self.setup_arrays()
    
    cpdef int output_fields(self) except -1:
        """
        Computes a scalar - the property value.
        """
        return 1

    cpdef setup_arrays(self):
        """
        """
        if self.source is None or self.dest is None:
            return

        SPHFunctionParticle3D.setup_arrays(self)

        self.d_prop = self.source.get_carray(self.prop_name)
        self.s_prop = self.source.get_carray(self.prop_name)

    cdef void eval(self, int source_pid, int dest_pid, KernelBase kernel, double
                   *nr, double *dnr):
        """
        """
        cdef double h=0.5*(self.s_h.data[source_pid] + 
                           self.d_h.data[dest_pid])
        
        cdef Point src_position = self._pnt1
        cdef Point dst_position = self._pnt2

        make_coords_3d(self.s_x, self.s_y, self.s_z, src_position, source_pid)
        make_coords_3d(self.d_x, self.d_y, self.d_z, dst_position, dest_pid)

        cdef double w = kernel.function(dst_position, src_position, h)
        cdef double temp = w*self.s_mass.data[source_pid]/(
            self.s_rho.data[source_pid])
        
        nr[0] += self.s_prop[source_pid]*temp
