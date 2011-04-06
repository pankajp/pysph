""" Tests for the solver """

import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph

import unittest
import numpy

Fluids = base.ParticleType.Fluid
Solids = base.ParticleType.Solid

class SolverTestCase(unittest.TestCase):
    """ A default solver test case.

    Setup:
    ------
    
    A dummy fluid solver is constructed to test the various functions 
    in solver/solver.py

    Tested Functions:
    -----------------
    
    (a) add_operation
    (b) replace_operation
    (c) remove operation
    (d) set_order
    (e) add_operation_step
    (f) setup_integrator
    (g) add_operation_xsph
    
    """
    def setUp(self):
        """ A Dummy fluid solver is created with the following operations

        (i)  -- Equation of State
        (ii) -- Density Rate
        (iii)-- Momentum Equation Pressure Gradient
        (iv) -- Momentum Equation Viscosity 
        
        """
        self.kernel = kernel = base.CubicSplineKernel(dim = 2)

        self.solver = s = solver.Solver(dim=2,
                                        integrator_type=solver.EulerIntegrator)

        s.default_kernel = kernel

        self.particles = base.Particles(arrays=[base.get_particle_array()])

        # Create some replacement operations
        
        self.visc = solver.SPHIntegration(

            sph.MorrisViscosity, on_types=[Fluids],
            from_types = [Fluids, Solids], updates=['u','v'], id='visc'
            
            )

        self.summation_density = solver.SPHOperation(

            sph.SPHRho, on_types=[Fluids, Solids],
            updates=['rho'], id='sd'

            )

    def setup_solver(self):

        s = self.solver

        # Add the default operations

        s.add_operation(solver.SPHOperation(
                
                sph.TaitEquation.withargs(1,1), on_types = [Fluids, Solids],
                updates=['p','cs'], id = 'eos')

                             )
                
        s.add_operation(solver.SPHIntegration(
                
                sph.SPHDensityRate, on_types=[Fluids, Solids],
                from_types=[Fluids, Solids], updates=['rho'], id='density_rate')
                             
                             )


        s.add_operation(solver.SPHIntegration(
                
                sph.SPHPressureGradient, on_types=[Fluids],
                from_types=[Fluids, Solids], updates=['u','v'], id='pgrad')

                             )

        s.add_operation(solver.SPHIntegration(

                sph.MonaghanArtificialVsicosity, on_types=[Fluids],
                from_types=[Fluids, Solids], updates=['u','v'], id='avisc')

                             )

    def test_constructor(self):
        """ Test for the construction of the solver """
        
        s = self.solver

        self.assertEqual(s.particles, None)
        
        self.assertEqual(s.default_kernel, self.kernel)

        self.assertEqual(s.operation_dict, {})
        
        self.assertEqual(s.order, [])

        self.assertEqual(s.t, 0.0)

        self.assertEqual(s.pre_step_functions, [])

        self.assertEqual(s.post_step_functions, [])

        self.assertEqual(s.pfreq, 100)

        self.assertEqual(s.dim, 2)
        
        self.assertEqual(s.kernel_correction, -1)

        self.assertEqual(s.pid, None)

        self.assertEqual(s.eps, -1)

    def test_setup_solver(self):
        """ Test setting up of the solver """
        
        self.setup_solver()

        s = self.solver

        order = s.order

        self.assertEqual(order, ['eos', 'density_rate', 'pgrad','avisc'])

    def test_remove_operation(self):
        """ Remove an operation """

        self.setup_solver()

        s = self.solver

        # Remove the avisc operation
        
        s.remove_operation('avisc')

        order = s.order

        self.assertEqual(order , ['eos', 'density_rate', 'pgrad'])

    def test_add_operation(self):
        
        self.setup_solver()

        s = self.solver

        # add a viscosity operation

        s.add_operation(self.visc)
        
        order = s.order
        
        self.assertEqual(order, ['eos','density_rate','pgrad','avisc','visc'])

        # remove the added operation

        s.remove_operation(self.visc)

        order = s.order

        self.assertEqual(order, ['eos','density_rate','pgrad','avisc'])

        # add the viscosity operation before avisc

        s.add_operation(self.visc, before=True, id='avisc')

        order = s.order
        
        self.assertEqual(order, ['eos','density_rate','pgrad', 'visc', 'avisc'])

        # remove the added operation

        s.remove_operation(self.visc)

        # add the viscosity operation after avisc

        s.add_operation(self.visc, before=False, id='avisc')

        order = s.order
        
        self.assertEqual(order, ['eos','density_rate','pgrad', 'avisc', 'visc'])


    def test_replace_operation(self):
        """ Nothing to write really! """
        
        self.setup_solver()

        s = self.solver

        s.replace_operation('avisc', self.visc)

        order = s.order

        self.assertEqual(order, ['eos','density_rate','pgrad', 'visc'])

    def test_step_operation(self):
        """ Test the stepping functions """

        self.setup_solver()
        
        s = self.solver

        s.add_operation_step(types=[Fluids])

        op = s.operation_dict['step']

        self.assertEqual(op.updates, ['x','y'])

        self.assertEqual(op.function.get_func_class(), sph.PositionStepping)

    def test_xsph_operation(self):
        """ Test for adding the XSPH function """

        self.setup_solver()

        s = self.solver

        s.add_operation_step(types=[Fluids])

        s.add_operation_xsph(eps = 0.5)

        op = s.operation_dict['xsph']

        self.assertEqual(op.on_types, [Fluids])
        
        self.assertEqual(op.from_types, [Fluids])

        self.assertEqual(op.updates, ['x','y'])

        self.assertEqual(op.function.get_func_class(), sph.XSPHCorrection)

    def test_setup_integrator(self):
        """ Test the setting up of the integrator """

        self.setup_solver()
        
        s = self.solver

        s.add_operation_step(types=[Fluids])

        s.add_operation_xsph(eps = 0.5)
        
        s.setup_integrator(self.particles)

        i = s.integrator

        calcs = i.calcs

        pcalcs = i.pcalcs

        icalcs = i.icalcs

        self.assertEqual(len(calcs), 6)

        self.assertEqual(len(pcalcs), 2)

if __name__ == '__main__':
    unittest.main()
