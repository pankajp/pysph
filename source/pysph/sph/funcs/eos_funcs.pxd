"""Declarations for the Equation of state SPH functions

"""

# Author: Prabhu Ramachandran <prabhu@aero.iitb.ac.in>
# Copyright (c) 2009, Prabhu Ramachandran

#sph imports
from pysph.sph.sph_func cimport SPHFunctionParticle

cdef class IdealGasEquation(SPHFunctionParticle):
    """ Ideal gas EOS """
			
    cdef public double gamma

cdef class TaitEquation(SPHFunctionParticle):
    """ Tait's equation of state """
    
    cdef public double gamma, co, ro


