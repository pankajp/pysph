"""
SPH functions for pressure related computations.
"""

from pysph.sph.sph_func cimport *
from pysph.base.carray cimport DoubleArray

cdef class SPHSymmetricPressureGradient3D(SPHFunctionParticle3D):
    """
    Computes pressure gradient using the formula 

        INSERTFORMULA

    """
    def __init__(self, ParticleArray source, ParticleArray dest,
                 str h='h', str mass='m', str rho='rho',
                 str coord_x='x', str coord_y='y', str coord_z='z',
                 str velx='u', str vely='v', str velz='z', bint
                 setup_arrays=True,
                 str pressure='p', 
                 *args, **kwargs):
        """
        Constructor.
        """
        SPHFunctionParticle3D.__init__(self, source, dest, h, mass, rho,
                                       coord_x, coord_y, coord_z,
                                       velx, vely, velz, setup_arrays=False)

        self.pressure = pressure

        if setup_arrays == True:
            self.setup_arrays()
        
    cpdef setup_arrays(self):
        """
        """
        if self.source is None or self.dest is None:
            return

        SPHFunctionParticle3D.setup_arrays(self)
        
        self.d_pressure = self.dest.get_carray(self.pressure)
        self.s_pressure = self.source.get_carray(self.pressure)

    cpdef int output_fields(self) except -1:
        """
        Returns 3 - the number of components this computes.
        """
        return 3

    cdef void eval(self, int source_pid, int dest_pid, KernelBase kernel, double
                   *nr, double *dnr):
        """
        """
        cdef double h = 0.5*(self.s_h.data[source_pid] +
                             self.d_h.data[dest_pid])
        cdef double temp = 0.0
        cdef Point grad = Point()

        make_coords_3d(self.s_x, self.s_y, self.s_z, self._pnt1, source_pid)
        make_coords_3d(self.d_x, self.d_y, self.d_z, self._pnt2, dest_pid)
        
        temp = self.s_pressure.data[source_pid]/(
            self.s_rho.data[source_pid]*self.s_rho.data[source_pid])
        
        temp += self.d_pressure[dest_pid]/(
            self.d_rho.data[dest_pid]*self.d_rho.data[dest_pid])

        temp *= self.s_mass.data[source_pid]
        
        kernel.gradient(self._pnt2, self._pnt1, h, grad)

        nr[0] += temp*grad.x
        nr[1] += temp*grad.y
        nr[2] += temp*grad.z
