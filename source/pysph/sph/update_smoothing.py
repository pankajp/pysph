"""
The ADKE algorithm to update the smoothing lengths works as follows:

for each particle array that must update it's smoothing length

    (a) Find a pilot estimate of the density
    (b) use the pilot estimate of the density to calculate scale factors
    (c) scale the smoothing lengths for the particles with the scale factors

The smoothing lengths are ideally updated after every sub step of an
integration step. That is, whenever the particles move and new
neighbors need to be calculated. After the smoothing lengths are
updated, we must recalculate the neighbors to reflect the modified
smoothing lengths.

The natural place to invoke the smoothing length calculation is in the
Particle's update function which is called whenever particles move (at
the end of a sub step)

"""

import numpy

class UpdateSmoothing:
    """ Base class for all update smoothing functions """

    def update_smoothing_lengths(self):
        pass

class UpdateSmoothingADKE(UpdateSmoothing):
    """ Update the smoothing length based on the ADKE algorithm:

    ..math::

    \overbar{\rho_a} = \sum_{b=a}^{b=N} m_b W(x_a-x_b, h0)

    log(g) = \frac{1}{N}\sum_{j=1}^{j=N}log(\overbar{\rho_a})
    
    \lambda_a = k(\frac{\overbar{\rho_a}}{g})^{-\epsilon}

    h_a = \lambda_a h0

    """

    def __init__(self, calcs, k = 1.0, eps=0.0, h0 = 1.0):
        """ Constructor.

         Parameters:
        -----------

        calcs -- The SPHCalc objects for which the smoothing lengths is updated.
        k -- numerical constant
        eps -- numerical constant
        h0 -- reference initial smoothing length for pilot rho estimate.

        Notes:
        ------

        The 'calcs' parameter should be all the calc objects returned
        via 'get_calcs' for the operation ADKEPilotRho

        """    
        
        self.calcs = calcs
        self.k = k
        self.eps = eps
        self.h0 = h0

        self._test_calcs()
        self.setup_calcs()

    def setup_calcs(self):

        for calc in self.calcs:
            dest = calc.dest

            if not dest.properties.has_key("pilotrho"):
                dest.add_property({"name":"pilotrho"})

            if not dest.properties.has_key("lam"):
                dest.add_property({"name":"lam"})

    def update_smoothing_lengths(self):

        ncalcs = len(self.calcs)
        for i in range(ncalcs):
            calc = self.calcs[i]
            nnps = calc.nnps_manager
            dest = calc.dest

            # set the pilot estimate for the calc

            calc.sph("pilotrho")
            
            # calculate g: log(g) = \frac{1}{N} \sum_{j=1}^{j=N} log(rhopilot_j)

            N = dest.num_real_particles
            pilotrho = dest.get("pilotrho")

            log_rho = numpy.log(pilotrho)
            log_g = numpy.sum(log_rho)
            log_g *= (1.0/N)

            g = numpy.exp(log_g)

            # \lambda_a = k (\frac{rhopilot_a}{g})^{-eps}

            _lambda = self.k * numpy.power(pilotrho/g, -self.eps)

            dest.set(lam = _lambda)

            # h_a = lambda_a * h0

            dest.set(h = _lambda*self.h0)

            # we need to recalculate the neighbors with the new h's 

            dest.set_dirty(True)
            nnps.py_update()            

    def _test_calcs(self):
        """ Test the calcs for valid functions.

        The function for each of the calcs should be the ADKEPilotRho
        function.

        """

        for calc in self.calcs:
            nsrcs = len(calc.sources)

            for i in range(nsrcs):
                func = calc.funcs[i]
                msg = "Invalid calc for UpdateSmoothingADKE"
                assert func.id == "pilotrho", msg
                
        
class TestUpdateSmoothingADKE(UpdateSmoothing):
    """ Test function to update the smoothing lengths.:

    A dummy ADKEUpdateSmoothing wherin the pilot estimate is not requested.
    The pilot estimate is taken from a default density distribution.

    """

    def __init__(self, calcs, h0=1.0, k=1.0, eps=0.0):
        """ Constructor.

         Parameters:
        -----------

        calcs -- The SPHCalc objects for which the smoothing lengths is updated.

        """    
        self.calcs = calcs
        self.k = k
        self.eps = eps
        self.h0 = h0

        self._test_calcs()
        self.setup_calcs()

    def setup_calcs(self):

        for calc in self.calcs:
            dest = calc.dest

            if not dest.properties.has_key("pilotrho"):
                dest.add_property({"name":"pilotrho"})

            if not dest.properties.has_key("lam"):
                dest.add_property({"name":"lam"})        


    def update_smoothing_lengths(self):

        ncalcs = len(self.calcs)
        for i in range(ncalcs):
            calc = self.calcs[i]
            nnps = calc.nnps_manager
            dest = calc.dest

            # calculate g: log(g) = \frac{1}{N} \sum_{j=1}^{j=N} log(rhopilot_j)

            N = dest.num_real_particles
            pilotrho = dest.get("rho")

            log_rho = numpy.log(pilotrho)
            log_g = numpy.sum(log_rho)
            log_g *= (1.0/N)

            g = numpy.exp(log_g)

            # \lambda_a = k (\frac{rhopilot_a}{g})^{-eps}

            _lambda = self.k * numpy.power(pilotrho/g, -self.eps)

            dest.set(lam = _lambda)

            # h_a = lambda_a * h0

            dest.set(h = _lambda*self.h0)

            # we need to recalculate the neighbors with the new h's 

            dest.set_dirty(True)
            nnps.py_update()            

    def _test_calcs(self):
        """ Test the calcs for valid functions.

        The function for each of the calcs should be the ADKEPilotRho
        function.

        """

        for calc in self.calcs:
            nsrcs = len(calc.sources)

            for i in range(nsrcs):
                func = calc.funcs[i]
                msg = "Invalid calc for UpdateSmoothingADKE"
                assert func.id == "pilotrho", msg
