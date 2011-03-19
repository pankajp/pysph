"""Declarations for the basic SPH functions 

"""
# Author: Prabhu Ramachandran <prabhu@aero.iitb.ac.in>
# Copyright (c) 2009, Prabhu Ramachandran

#sph imports
from pysph.sph.sph_func cimport SPHFunction

cdef class PostionStepping(SPHFunction):
    pass
