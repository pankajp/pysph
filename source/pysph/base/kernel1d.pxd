"""
1D kernel declararions.
"""
# local imports
from pysph.base.kernelbase cimport Kernel1D


################################################################################
# `Lucy1D` class.
################################################################################ 
cdef class Lucy1D(Kernel1D):
    """
    This class represents Lucy's (1977) kernel.
    """
    pass

################################################################################
# `Gaussian1D` class.
################################################################################ 
cdef class Gaussian1D(Kernel1D):
    """
    This class represents a 1D Gaussian kernel.
    """
    pass

################################################################################
# `Cubic1D` class.
################################################################################ 
cdef class Cubic1D(Kernel1D):
    """Cubic B-spline kernel in 1D."""
    pass
