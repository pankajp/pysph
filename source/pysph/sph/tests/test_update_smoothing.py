""" Test for functions in update_smoothing lengths """


import pysph.base.api as base
import pysph.sph.api as sph
import pysph.solver.api as solver

import numpy
import unittest

Fluid = base.ParticleType.Fluid

class UpdateSmoothingTestCase(unittest.TestCase):
    """ Default setup to test for updating the smoothing lengths in 1D

    Setup:
    ------

    The setup consists of 11 equispaced particles in the interval
    [0,1] with the following properties:

    dx = 0.1, h = 0.1, m = 0.1, h0 = 0.1, rho = 1.0

    The kernel used is the CubicSplineKernel which has a radius scale
    factor of 2. This means only two adjacent particles on
    either side appear as neighbors with the default 'h'    

    """

    def setUp(self):
        """
        Setup:
        ------
        
        The setup consists of 11 equispaced particles in the interval
        [0,1] with the following properties:
        
        dx = 0.1, h = 0.1, m = 0.1, h0 = 0.1, rho = 1.0
        
        The kernel used is the CubicSplineKernel which has a radius scale
        factor of 2. This means only two adjacent particles on
        either side appear as neighbors with the default 'h'    
    
        """
        
        x = numpy.linspace(0,1,11)
        dx = x[1] - x[0]
        h = numpy.ones_like(x) * (dx+1e-10)
        m = numpy.ones_like(x) * dx
        rho = numpy.ones_like(x)
        h0 = dx

        pa = base.get_particle_array(name="test", type=Fluid,
                                     x=x, h=h, rho=rho, m=m, tmp=m)

        particles = base.Particles(arrays=[pa], variable_h=True)

        # smoothing length update operation

        adke = solver.SPHOperation(
            sph.ADKESmoothingUpdate.withargs( h0=h0, k=1.0, eps=0.0 ),
            from_types=[Fluid], on_types=[Fluid],
            updates=["h"],
            id="adke_smoothing")

        kernel = base.CubicSplineKernel(dim=1)

        calcs = adke.get_calcs(particles, kernel)

        self.h0 = h0
        self.calcs = calcs
        self.particles = particles

    def test_particles_update(self):
        """ Test the particle's update function.

        New neighbor information is sought when the particles move or
        their smoothing lengths change.

        In this test, the smoothing length of particle '5' is doubled
        and neighbors are queried.

        This test essentially ensures that explicitly setting the
        dirty flag of the particle array will ensure a new neighbor
        querry (provided particles.update is called!)

        """

        particles = self.particles

        loc = self.calcs[0].nbr_locators[0]
        pa = particles.get_named_particle_array("test")

        nbrs = loc.py_get_nearest_particles(5)
        nbrs = nbrs.get_npy_array()
        nbrs.sort()

        _nbrs = [3,4,5,6,7]

        self.assertEqual(len(nbrs), len(_nbrs))

        for i in range(len(nbrs)):
            self.assertEqual(nbrs[i], _nbrs[i])


        # artificially double the smoothing length of particle 5

        h = pa.get('h')

        h[5] *= 2

        pa.set_dirty(True)

        # update the particles

        particles.update()

        nbrs = loc.py_get_nearest_particles(5)
        nbrs = nbrs.get_npy_array()
        nbrs.sort()

        _nbrs = [1,2,3,4,5,6,7,8,9]

        self.assertEqual(len(nbrs), len(_nbrs))
        for i in range(len(nbrs)):
            self.assertEqual(nbrs[i], _nbrs[i])

    def test_adke_default(self):
        """ Test the ADKE algorithm with the default parameters of

        h0 = 1.0, k = 1.0, eps = 0.0

        The smoothing lengths should remain unchanged with these parameters.

        """

        particles = self.particles
        pa = particles.get_named_particle_array("test")
        np = pa.get_number_of_particles()

        _h = pa.get("h").copy()

        for i in range(np):
            self.assertAlmostEqual(_h[i], 0.1, 8)

        # perform the update

        particles.update()

        h = pa.get('h')

        for i in range(np):
            self.assertAlmostEqual(h[i], _h[i], 8)
        
class ADKETstCase(UpdateSmoothingTestCase):
    """ Test the ADKE Algorithm

    We use the TestADKEUpdateSmoothing function to use a pilot density
    estimate that is set manually.

    The density distribution is

    rho = 1, 1, 1, 1, 1, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25

    With this distribution, the ADKE parameters are

    log_g = (1.0/11) * numpy.sum(numpy.log(rho)) =  -0.75616056061084935

    g = 0.46946545533085321

    """

    def test_adke(self):
        """ Test the ADKE algorithm with varying parameter values. """

        particles = self.particles
        pa = particles.get_named_particle_array("test")

        # set the density distribution

        rho = pa.get("rho")
        rho[5:] = 0.25
        pa.set(rho=rho)

        # ADKE: k = 1.0, eps = 1.0

        particles.update()

class TestUpdateSmoothing(sph.SPHFunctionParticle):
    """ Test function to update the smoothing lengths:

    The smoothing length of the fifth particle is doubled i.e. h[5] *= 2

    """
    def __init__(self, source, dest, setup_arrays=True):
        sph.SPHFunctionParticle.__init__(self, source, dest, setup_arrays)
        self.num_outputs = 1
        self.dst_reads = ['h']

    def eval(self, kernel, output1, output2, output3):
        for i in range(output1.length):
            output1[i] = self.d_h[i]

        output1[5] *= 2

class TestUpdateSmoothingSolver(unittest.TestCase):
    def setUp(self):
        x = numpy.linspace(0,1,11)
        dx = x[1] - x[0]
        h = numpy.ones_like(x) * (dx+1e-10)
        m = numpy.ones_like(x) * dx
        rho = numpy.ones_like(x)
        self.h0 = dx

        self.pa = pa = base.get_particle_array(name="test", type=Fluid,
                                     x=x, h=h, rho=rho, m=m, tmp=m)

        self.particles = base.Particles(arrays=[pa], variable_h=True)

        # smoothing length update operation

        self.adke = solver.SPHOperation(
            TestUpdateSmoothing,
            from_types=[Fluid], on_types=[Fluid],
            updates=["tmp"], # updates='h' will first reset array h to 0
            id="pilotrho" )
        self.kernel = base.CubicSplineKernel(dim=1)
    
    def test_solve(self):

        for integrator in [solver.EulerIntegrator]:

            self.solver = solver.Solver(self.kernel.dim, integrator)
            self.solver.add_operation(self.adke)
            self.solver.setup_solver()
            self.solver.setup_integrator(self.particles)
            self.solver.set_time_step(1e-4)
            self.solver.set_final_time(1e-4)
            
            # eval is called nstep number of times for each integrator
            # and doubles the fifth particle's h each time
            
            h_expect = self.pa.h[5] * 2**self.solver.integrator.nsteps

            self.solver.solve(False)
            self.assertAlmostEqual(self.pa.tmp[5], h_expect)
    
if __name__ == "__main__":
    unittest.main()
