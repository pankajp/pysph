from pysph.base.carray cimport DoubleArray

cdef class KernelCorrection:
    cdef evaluate_correction_terms(self)

    cdef set_correction_terms(self, calc)

cdef class BonnetAndLokKernelCorrection(KernelCorrection):
    """ Bonnet and Lok corection """
    
    cdef public int dim 
    cdef public int np
    cdef public object calc
    
    cdef public DoubleArray bl_l11, bl_l12, bl_l13, bl_l22, bl_l23, bl_l33

cdef class KernelCorrectionManager:
    """ Correction Manager """
    
    cdef public dict correction_functions
    cdef public list calcs
    cdef public int kernel_correction

    cdef cache_kernel_correction_functions(self)

    cdef set_correction_terms(self, calc)

    cpdef update(self)
