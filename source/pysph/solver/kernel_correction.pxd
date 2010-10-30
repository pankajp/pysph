from pysph.sph.sph_calc cimport SPHBase
from pysph.base.carray cimport DoubleArray


cdef class BonnetAndLokKernelCorrection:
    """ Bonnet and Lok corection """
    
    cdef public SPHBase calc
    cdef public int dim 
    
    cdef DoubleArray bl_l11, bl_l12, bl_l13, bl_l22, bl_l23, bl_l33
    

    cdef evaluate_correction_terms(self)
