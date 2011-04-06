"""  Tests for the sph_equation function and clases """

import pysph.base.api as base
import pysph.sph.api as sph
import pysph.solver.api as solver
import numpy, unittest

Fluid = base.ParticleType.Fluid
Solid = base.ParticleType.Solid

class SPHOperationTestCase(unittest.TestCase):
    """ SPHOperation is responsible for returning the appropriate 
    calc setup data for the various subclasses. 

    The calc data performs the filtering of the input arrays based on
    the type and the appropriate setting up of the function and
    sources for a particular calc. Thus, the main purpose of this test is
    to test the get_calc_data function 

    """

    def setUp(self):
        """ Setup for SPHOperationTestCase

        Setup:
        ------
        Create two particle arrays, one Fluid and one Solid.
        Instantiate the class with a default function `SPHRho` with 
        various combinations of the from and on types and check for the 
        filtering of the arrays.

        """

        x = numpy.linspace(0,1,11)
        h = numpy.ones_like(x) * 2 * (x[1] - x[0])

        #create the fluid particle array

        self.fluid = base.get_particle_array(name='fluid', type=Fluid, x=x,h=h)

        #create the solid particle array
        
        self.solid = base.get_particle_array(name="solid", type=Solid, x=x,h=h)

        #create the particles
        
        self.particles = particles = base.Particles(arrays=[self.fluid,
                                                            self.solid])

        #set the kernel
        
        self.kernel = base.CubicSplineKernel()

    def test_from_solid_on_fluid(self):
        """ Test for construction of the SPHOperation """

        function = sph.SPHRho
        operation = solver.SPHOperation(

            function, on_types=[Fluid], updates=['rho'], id = 'sd',
            from_types=[Fluid, Solid]

            )
        
        #test for the construction
        self.assertEqual(type(operation.function), type(function))
        self.assertEqual(operation.from_types, [Fluid, Solid])
        self.assertEqual(operation.on_types, [Fluid])
        self.assertEqual(operation.updates, ['rho'])
        self.assertEqual(operation.id, 'sd')        

        calc_data = operation.get_calc_data(self.particles)

        # test the calc data

        # one destination for type Fluid

        ndsts = len(calc_data)
        self.assertEqual(ndsts, 1)

        # the destination number should be 0 ( the first array is Fluid )
        
        dest_data = calc_data[0]
        self.assertEqual(dest_data['dnum'], 0)

        # the calc should have two sources

        sources = dest_data['sources']
        self.assertEqual(len(sources), 2)

        # the calc should have two funcs ( one for each src-dst pair )

        funcs = dest_data['funcs']
        self.assertEqual(len(funcs), 2)
        
        # test the first function
        
        func = funcs[0]        
        self.assertEqual(func.dest, self.fluid)
        self.assertEqual(func.source, self.fluid)

        #test the second function
        
        func = funcs[1]        
        self.assertEqual(func.dest, self.fluid)
        self.assertEqual(func.source, self.solid)

############################################################################
class SPHIntegrationTestCase(SPHOperationTestCase):
    """ Test for the calc returned by integrating SPHIntegration """

    def test_calc(self):
        
        function = sph.GravityForce.withargs(gy=-9.81)
        operation = solver.SPHIntegration(
            
            function=function, on_types=[Solid,Fluid],
            updates=['u', 'v'], id='gravity'
        
            )

        calcs = operation.get_calcs(particles=self.particles, 
                                    kernel=self.kernel)

        ncalcs = len(calcs)
        self.assertEqual(ncalcs, 2)

        # Test for the first calc
        # dnum, dest = (0, fluid)

        calc1 = calcs[0]
        self.assertEqual(calc1.dnum, 0)
        self.assertEqual(calc1.dest, self.fluid)

        # nsrcs = 0 

        sources = calc1.sources
        nsrcs = len(sources)
        self.assertEqual(nsrcs, 0)

        # integrates == True

        self.assertEqual(calc1.integrates, True)
        self.assertEqual(calc1.nbr_info, True)

        # updates = ['u','v']
        
        updates = calc1.updates
        nupdates = len(calc1.updates)

        self.assertEqual(nupdates, 2)
        self.assertEqual(updates, ['u','v'])

        # Test for the second calc
        # dnum, dest = (0, solid)

        calc = calcs[1]
        self.assertEqual(calc.dnum, 1)
        self.assertEqual(calc.dest, self.solid)

        # nsrcs = 0
        
        sources = calc.sources
        nsrcs = len(sources)
        self.assertEqual(nsrcs, 0)

        # integrates = True

        self.assertEqual(calc.integrates, True)
        self.assertEqual(calc.nbr_info, True)

        # updates = ['u','v']

        updates = calc.updates
        nupdates = len(calc.updates)
        self.assertEqual(nupdates, 2)
        self.assertEqual(updates, ['u','v'])
        
############################################################################

if __name__ == '__main__':
    unittest.main()
