""" Funcitons to handle the kernel correction """

from pysph.base.particle_array cimport ParticleArray
from pysph.sph.sph_func cimport SPHFunctionParticle
from pysph.base.particle_tags cimport LocalReal, Dummy
from pysph.base.carray cimport LongArray
from pysph.base.nnps cimport FixedDestNbrParticleLocator
from pysph.sph.funcs.basic_funcs cimport BonnetAndLokKernelGradientCorrectionTerms

cdef class BonnetAndLokKernelCorrection:
    """ Bonnet and Lok corection """

    def __init__(self, calc):

        self.calc = calc
        self.dim = calc.dim
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

    cdef evaluate_correction_terms(self):
        """ Perform the kernel correction """

        cdef SPHBase calc = self.calc
        cdef ParticleArray dest = calc.dest
        
        cdef SPHFunctionParticle func
        cdef FixedDestNbrParticleLocator loc
        cdef ParticleArray src

        cdef long dest_pid, source_pid
        cdef double nr[3], dnr[3]
        cdef int i, np, k, s_idx
        cdef int j

        cdef LongArray nbrs = self.nbrs

        np = dest.get_number_of_particles()


        cdef LongArray tag_arr = self.dest.get_carray('tag')
        cdef long* tag = tag_arr.get_data_ptr()

        for i in range(np):
            
            if tag[i] == LocalReal:

                dnr[0] = dnr[1] = dnr[2] = 0.0
                nr[0] = nr[1] = nr[2] = 0.0

                for j in range(calc.nsrcs):

                    src = calc.sources[j]
                    loc = calc.nbr_locators[j]                    
            
                    nbrs.reset()
                    #loc.get_nearest_particles(i, nbrs, exclude_self)

                    correction_func=BonnetAndLokKernelGradientCorrectionTerms(
                        source=src, dest=dest)

                    for k from 0 <= k < self.nbrs.length:
                        s_idx = self.nbrs.get(k)

                        #corection_func.eval(s_idx, i, calc.kernel, 
                        #                    &nr[0], &dnr[0])


class KernelCorrectionManager(object):
    
    def __init__(self, calcs, kernel_correction=0):
        """ Manage all correction related functions

        Parameters:
        -----------
        calcs -- the calc objects in a simulation.
        kernel_correction -- the type of kernel correction to use.

        """
        
        self.correction_functions = {}

        #store only those calcs for which neighbor information is required

        for i, calc in enumerate(calcs):
            if not calc.nbr_info:
                calcs.pop(i)
                
        #store the type of kernel correction to use

        self.kernel_correction = kernel_correction

        #cache the kernel correction functions

        self.cache_kernel_correction_functions()

    def cache_kernel_correction_functions(self):
        """ Cache the kernel correction functions """

        calcs = self.calcs

        func = BonnetAndLokKernelCorrection

        for calc in calcs:
            if not self.correction_functions.has_key(calc.snum):
                self.correction_functions[calc.snum] = func(calc)            
        
    def get_correction_function(self, calc):
        """ Return a cached correction function """
        return self.correction_functions[calc.snum]

    def evaluate_correction_terms(self, calc):
        """ Evaluate the correction terms """
        
        func = self.get_correction_function(calc)
        func.evaluate_correction_terms()
