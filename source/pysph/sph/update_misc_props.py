import numpy

class UpdateDivergence:
    """ Update the conduction coefficient for use in the energy equation

    The equation added to the energy equation is:

    ..math::
    \frac{1}{\rho}\,\nabla \cdot (q\nabla\,U)

    where, U is the specific thermal energy and q the conduction coefficient 

    q is given by (Sigalotti[2006]):

    ..math::
    q_a = g1 h_a c_a + g2 h_a^2 [abs(div)_a - div_a]

    """

    def __init__(self, calcs):
        """ Constructor

        Parameters:
        -----------

        calcs -- The SPHCalc objects for which 'q' is required
        g1 -- numerical constant
        g2 -- numerical constant

        Notes:
        ------

        The 'calcs' parameter should be the list of calcs returned via
        'get_calcs' from the operation of divergence. 

        """    
        self.calcs = calcs

        self._test_calcs()
        self.setup_calcs()

    def setup_calcs(self):

        for calc in self.calcs:
            dest = calc.dest

            if not dest.properties.has_key("q"):
                dest.add_property({"name":"q"})

            if not dest.properties.has_key("div"):
                dest.add_property({"name":"div"})

    def eval(self):

        ncalcs = len(self.calcs)
        for i in range(ncalcs):
            calc = self.calcs[i]
            dest = calc.dest

            # set the divergence for the dest

            calc.sph("div")
            
            # calculate q: q_a = g1 h_a c_a + g2 h_a^2 [abs(div_a) - div_a]

            #ha, ca, diva = dest.get("h", "cs", "div")
            
            #q = self.g1 * ca + g2 * ha * (numpy.abs(diva) - diva)
            #q *= ha

            #dest.set(q = q)

    def _test_calcs(self):
        """ Test the calcs for valid functions.

        The calculation of the conduction coefficient requires the
        divergence of velocity. The calcs provided to this class must
        implement the velocity divergence function defined in
        sph/funcs/adke_funcs.pyx

        """

        for calc in self.calcs:
            nsrcs = len(calc.sources)

            for i in range(nsrcs):
                func = calc.funcs[i]
                assert func.id == "vdivergence", "Invalid calc!"

