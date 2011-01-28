"""
Tests for the sph_calc module.
"""
# standard imports
import unittest
import numpy

import pysph.base.api as base
import pysph.sph.api as sph

Fluid = base.ParticleType.Fluid
Solid = base.ParticleType.Solid

def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-16."""
    return numpy.allclose(x, y, atol=1e-16, rtol=0)

##############################################################################
        
if __name__ == '__main__':
    unittest.main()
