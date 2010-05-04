""" Definitions for the SPH summations defined in the paper: "SPH and
Riemann Sovers" by J.J. Monaghan, JCP, 136, 298-307. """

# Author: Kunal Puri <kunalp@aero.iitb.ac.in>
# Copyright (c) 2010, Kunal Puri.

from pysph.base.carray cimport DoubleArray
from pysph.sph.sph_func cimport SPHFunctionParticle3D
from pysph.base.point cimport Point
from pysph.base.kernels cimport MultidimensionalKernel
from pysph.base.particle_array cimport ParticleArray


############################################################################
#`PressureGradient` class
############################################################################
cdef class PressureGradient(SPHFunctionParticle3D):
    """ 
    The gradient of pressure in the momentum equation.

    """
    #String literal for the pressure
    cdef public str p

    #The source and destination arrays for pressure
    cdef public DoubleArray s_p, d_p

    #The parameter sigma
    cdef public double sigma

############################################################################
#`MomentumEquationAvisc` class
############################################################################
cdef class MomentumEquationAvisc(SPHFunctionParticle3D):
    """ 
    Definitions for the artificial viscosity contribution to the 
    momentum equation 
    """
    #String literal to identify the pressure array
    cdef public str pressure
    
    #The arrays for the source and destination pressures.
    cdef public DoubleArray s_pressure, d_pressure
    
    #The unit vector between a pair of particles.
    cdef public Point j

    #The signal velocity between a pair of particles.
    cdef public Point vsig 

    #The coefficient `K`
    cdef public double k

    #The ratio of specific heats
    cdef public double gamma

    #Constant for the signal velocity
    cdef public double beta
############################################################################

############################################################################
#`EnergyEquation` class
############################################################################
cdef class EnergyEquation(SPHFunctionParticle3D):
    """
    Definitions for the contribution to the thermal energy equation 
    """    
    #String literal to identify the pressure and energy
    cdef public str pressure, energy
    
    #The arrays for the source and destination pressures.
    cdef public DoubleArray s_pressure, d_pressure, s_energy, d_energy

    #The unit vector between a pair of particles.
    cdef public Point j

    #The signal velocity between a pair of particles.
    cdef public Point vsig 
    
    #The coefficient `K`
    cdef public double k

    #The ratio of specific heats
    cdef public double gamma

    #Constant for the signal velocity
    cdef public double beta

    #Scaling constants for the thermal energy
    cdef public double f

############################################################################

############################################################################
#`EnergyEquationAvisc` class
############################################################################
cdef class EnergyEquationAvisc(SPHFunctionParticle3D):
    """ 
    Definitions for the artificial viscosity contribution to the 
    energy equation 
    """
    #The unit vector between a pair of particles.
    cdef public Point j

    #The signal velocity between a pair of particles.
    cdef public Point vsig 

    #The coefficient `K`
    cdef public double k

    #The ratio of specific heats
    cdef public double gamma

    #Constant for the signal velocity
    cdef public double beta
############################################################################
