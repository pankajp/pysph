""" Funcitons to handle the kernel correction """

from pysph.base.particle_array cimport ParticleArray, LocalReal, Dummy
from pysph.sph.sph_func cimport SPHFunctionParticle
from pysph.base.carray cimport LongArray
from pysph.base.nnps cimport FixedDestNbrParticleLocator

from pysph.sph.funcs.basic_funcs cimport BonnetAndLokKernelGradientCorrectionTerms

cdef extern from "math.h":
    double sqrt(double)
    double fabs(double)

cdef class KernelCorrection:
    """ Base class for kernel correction """

    cdef evaluate_correction_terms(self):
        raise NotImplementedError

    cdef set_correction_terms(self, calc):
        raise NotImplementedError

cdef class BonnetAndLokKernelCorrection(KernelCorrection):
    """ Bonnet and Lok correction """

    def __init__(self, calc):

        self.calc = calc
        self.dim = calc.dim
        self.np = calc.dest.get_number_of_particles()
        self.setup()

    def setup(self):
        
        calc = self.calc
        
        #setup the arrays required by the destination particle array

        dest = calc.dest

        if not dest.properties.has_key("bl_l11"):
            dest.add_property({"name":"bl_l11"})

        if not dest.properties.has_key("bl_l12"):
            dest.add_property({"name":"bl_l12"})

        if not dest.properties.has_key("bl_l13"):
            dest.add_property({"name":"bl_l13"})

        if not dest.properties.has_key("bl_l22"):
            dest.add_property({"name":"bl_l22"})

        if not dest.properties.has_key("bl_l23"):
            dest.add_property({"name":"bl_l23"})

        if not dest.properties.has_key("bl_l33"):
            dest.add_property({"name":"bl_l33"})

        for i in range(calc.nsrcs):
            func = calc.funcs[i]

            func.bl_l11 = dest.get_carray("bl_l11")

            func.bl_l12 = dest.get_carray("bl_l12")

            func.bl_l13 = dest.get_carray("bl_l13")

            func.bl_l22 = dest.get_carray("bl_l22")

            func.bl_l23 = dest.get_carray("bl_l23")

            func.bl_l33 = dest.get_carray("bl_l33")

        self.bl_l11 = DoubleArray(self.np)
        self.bl_l12 = DoubleArray(self.np)
        self.bl_l13 = DoubleArray(self.np)
        self.bl_l22 = DoubleArray(self.np)
        self.bl_l23 = DoubleArray(self.np)
        self.bl_l33 = DoubleArray(self.np)

    def resize_arrays(self):
        """ Resize the arrays """
        dest = self.calc.dest
        self.np = dest.get_number_of_particles()

        self.bl_l11.resize(self.np)
        self.bl_l12.resize(self.np)
        self.bl_l13.resize(self.np)
        self.bl_l22.resize(self.np)
        self.bl_l23.resize(self.np)
        self.bl_l33.resize(self.np)

    cdef evaluate_correction_terms(self):
        """ Perform the kernel correction """

        calc = self.calc
        cdef ParticleArray dest = calc.dest
        
        cdef SPHFunctionParticle func, cfunc
        cdef FixedDestNbrParticleLocator loc
        cdef ParticleArray src

        cdef long dest_pid, source_pid
        cdef double nr[3], dnr[3]
        cdef int i, np, k, s_idx
        cdef int j

        cdef double m11, m12, m13, m22, m23, m33, m21, m31, m32

        cdef double m33m22, m32m23, m33m12, m32m13, m23m12, m22m13

        cdef double m33m21, m31m23, m33m11, m31m13, m23m11, m21m13
        
        cdef double m32m21, m31m22, m32m11, m31m12, m22m11, m21m12
        
        cdef double det

        cdef LongArray nbrs = calc.nbrs

        np = dest.get_number_of_particles()

        cdef LongArray tag_arr = dest.get_carray('tag')
        cdef long* tag = tag_arr.get_data_ptr()

        for i in range(np):

            dnr[0] = dnr[1] = dnr[2] = 0.0
            nr[0] = nr[1] = nr[2] = 0.0
                
            if tag[i] == LocalReal:                

                for j in range(calc.nsrcs):

                    src = calc.sources[j]
                    loc = calc.nbr_locators[j]
            
                    nbrs = loc.get_nearest_particles(i, False)

                    cfunc=BonnetAndLokKernelGradientCorrectionTerms(
                        source=src, dest=dest)

                    for k from 0 <= k < nbrs.length:
                        s_idx = nbrs.get(k)
                        cfunc.eval(k, s_idx, i, calc.kernel, &nr[0], &dnr[0])

                m11 = nr[0]; m12 = nr[1]; m13 = nr[2]
                m22 = dnr[0]; m23 = dnr[1]; m33 = dnr[2]

                m21 = m12; m31 = m13; m32 = m23

                if self.dim == 1:
                    m22 = 1.0; m33 = 1.0

                if self.dim == 2:
                    m33 = 1.0

                m33m22 = m33*m22; m32m23 = m32*m23; m33m12 = m33*m12
                m32m13 = m32*m13; m23m12 = m23*m12; m22m13 = m22*m13

                m33m21 = m33*m21; m31m23 = m31*m23; m33m11 = m33*m11
                m31m13 = m31*m13; m23m11 = m23*m11; m21m13 = m21*m13

                m32m21 = m32*m21; m31m22 = m31*m22; m32m11 = m32*m11
                m31m12 = m31*m12; m22m11 = m22*m11; m21m12 = m21*m12

                det = m11*(m33m22 - m32m23)
                det -= m21*(m33m12 - m32m13)
                det += m31*(m23m12 - m22m13)

                if fabs(det) > 0.01 and fabs(m11) > 0.25 and fabs(m22) > 0.25 \
                        and fabs(m33) > 0.25:

                    self.bl_l11.data[i] = 1./det * (m33m22-m32m23)
                    self.bl_l12.data[i] = 1./det * -(m33m12-m32m13)
                    self.bl_l13.data[i] = 1./det * (m23m12-m22m13)
                    self.bl_l22.data[i] = 1./det * (m33m11-m31m13)
                    self.bl_l23.data[i] = 1./det * -(m23m11-m21m13)
                    self.bl_l33.data[i] = 1./det * (m22m11-m21m12)

                else:
                    self.bl_l11.data[i] = 1.0
                    self.bl_l12.data[i] = 0.0
                    self.bl_l13.data[i] = 0.0
                    self.bl_l22.data[i] = 1.0
                    self.bl_l23.data[i] = 0.0
                    self.bl_l33.data[i] = 1.0

    cdef set_correction_terms(self, calc):
        """ Set the correction terms for the calc's func """

        cdef SPHFunctionParticle func
        cdef ParticleArray dest

        cdef DoubleArray f_l11, f_l12, f_l13, f_l22, f_l23, f_l33
        cdef DoubleArray s_l11, s_l12, s_l13, s_l22, s_l23, s_l33

        cdef int i, np, j

        dest = calc.dest
        np = dest.get_number_of_particles()

        for i in range(calc.nsrcs):
            func = calc.funcs[i]

            f_l11 = func.bl_l11; f_l12 = func.bl_l12; f_l13 = func.bl_l13
            f_l22 = func.bl_l22; f_l23 = func.bl_l23; f_l33 = func.bl_l33

            s_l11 = self.bl_l11; s_l12 = self.bl_l12; s_l13 = self.bl_l13
            s_l22 = self.bl_l22; s_l23 = self.bl_l23; s_l33 = self.bl_l33
            
            for j in range(np):
                f_l11.set(j, s_l11.get(j))

                f_l12.set(j, s_l12.get(j))

                f_l13.set(j, s_l13.get(j))

                f_l22.set(j, s_l22.get(j))

                f_l23.set(j, s_l23.get(j))

                f_l33.set(j, s_l33.get(j))

    def py_evaluate_correction_terms(self):
        self.evaluate_correction_terms()

    def py_set_correction_terms(self, calc):
        self.set_correction_terms(calc)

cdef class KernelCorrectionManager(object):
    
    def __init__(self, calcs, kernel_correction=-1):
        """ Manage all correction related functions

        Parameters:
        -----------
        calcs -- the calc objects in a simulation.
        kernel_correction -- the type of kernel correction to use.

        """
        
        self.correction_functions = {}

        #store only those calcs for which neighbor information is required

        self.calcs = []

        for i, calc in enumerate(calcs):
            calc.correction_manager = self
            if calc.nbr_info:
                self.calcs.append(calc)
                
        #store the type of kernel correction to use

        self.kernel_correction = kernel_correction

        #cache the kernel correction functions

        if self.kernel_correction != -1:
            self.cache_kernel_correction_functions()

    cdef cache_kernel_correction_functions(self):
        """ Cache the kernel correction functions """

        cdef list calcs = self.calcs

        cfunc = BonnetAndLokKernelCorrection

        for calc in calcs:
            if not self.correction_functions.has_key(calc.snum):
                self.correction_functions[calc.snum] = cfunc(calc)

            else:
                dest = calc.dest
                for i in range(calc.nsrcs):
                    func = calc.funcs[i]

                    func.bonnet_and_lok_correction=True

                    func.bl_l11 = dest.get_carray("bl_l11")

                    func.bl_l12 = dest.get_carray("bl_l12")
                    
                    func.bl_l13 = dest.get_carray("bl_l13")
                    
                    func.bl_l22 = dest.get_carray("bl_l22")
                    
                    func.bl_l23 = dest.get_carray("bl_l23")
                    
                    func.bl_l33 = dest.get_carray("bl_l33")
        
    cdef set_correction_terms(self, calc):
        """ Evaluate the correction terms """

        cdef KernelCorrection cfunc
        
        cfunc = self.correction_functions[calc.snum]
        cfunc.set_correction_terms(calc)

    cpdef update(self):
        """ Evaluate the correction terms when particle config changes """

        cdef KernelCorrection cfunc
        cdef str id

        if self.kernel_correction != -1:
            for id, cfunc in self.correction_functions.iteritems():
                cfunc.resize_arrays()
                cfunc.evaluate_correction_terms()
        
    def py_set_correction_terms(self, calc):
        self.set_correction_terms(calc)
    
