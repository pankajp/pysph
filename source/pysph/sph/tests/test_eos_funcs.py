""" Tests for the eos functions """

import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph

if solver.HAS_CL:
    import pyopencl as cl

import numpy
import unittest
from os import path

NSquareLocator = base.NeighborLocatorType.NSquareNeighborLocator

class EOSFunctionsTestCase(unittest.TestCase):

    def runTest(self):
        pass

    def setUp(self):
        """ The setup consists of four particles placed at the
        vertices of a unit square.

        The function tested is

        ..math::

        p_a = (\gamma - 1.0)\rho_a U_a
        cs_a = \sqrt( (\gamma - 1.0) U_a )


        """
        
        self.precision = "single"

        self.np = 4

        x = numpy.array([0, 0, 1, 1], numpy.float64)
        y = numpy.array([0, 1, 1, 0], numpy.float64)

        z = numpy.zeros_like(x)
        m = numpy.ones_like(x)

        e = numpy.array([1, 2, 1, 2], numpy.float64)
        rho = numpy.array([2, 1, 2, 1], numpy.float64)
        
        tmpx = numpy.zeros_like(x)
        tmpy = numpy.zeros_like(x)
        tmpz = numpy.zeros_like(x)

        self.pa = pa = base.get_particle_array(name="test", x=x,  y=y, z=z,
                                               m=m, e=e, rho=rho,
                                               tmpx=tmpx, tmpy=tmpy, tmpz=tmpz,
                                               cl_precision=self.precision)

        ideal = sph.IdealGasEquation.withargs(gamma=1.4)

        self.ideal = ideal.get_func(pa,pa)
        
        self.ideal.nbr_locator = \
                               base.Particles.get_neighbor_particle_locator(pa,
                                                                            pa)

        self.setup_cl()

    def setup_cl(self):
        pass

class IdealGasEquationTestCase(EOSFunctionsTestCase):

    def setup_cl(self):
        pa = self.pa
        
        if solver.HAS_CL:
            self.ctx = ctx = cl.create_some_context()
            self.q = q = cl.CommandQueue(ctx)

            pa.setup_cl(ctx, q)
            
            pysph_root = solver.get_pysph_root()
            
            template = solver.cl_read(
                path.join(pysph_root, "sph/funcs/eos_funcs.clt"),
                function_name=self.ideal.cl_kernel_function_name,
                precision=self.precision)

            prog_src = solver.create_program(template, self.ideal)

            self.prog=cl.Program(ctx, prog_src).build(solver.get_cl_include())

    def get_reference_solution(self):
        """ Evaluate the force on each particle manually """
        
        pa = self.pa
        result = []

        rho, e = pa.get('rho', 'e')

        kernel = base.CubicSplineKernel(dim=2)
        gamma = 1.4

        for i in range(self.np):

            force = base.Point()

            rhoa = rho[i]
            ea = e[i]

            force.x = (gamma - 1.0) * ea * rhoa
            force.y = numpy.sqrt( (gamma-1.0) * ea )

            result.append(force)

        return result

    def test_eval(self):
        """ Test the PySPH solution """

        pa = self.pa
        func = self.ideal

        k = base.CubicSplineKernel(dim=2)

        tmpx = pa.properties['tmpx']
        tmpy = pa.properties['tmpy']
        tmpz = pa.properties['tmpz']        

        func.eval(k, tmpx, tmpy, tmpz)

        reference_solution = self.get_reference_solution()

        for i in range(self.np):
            self.assertAlmostEqual(reference_solution[i].x, tmpx[i])
            self.assertAlmostEqual(reference_solution[i].y, tmpy[i])
            self.assertAlmostEqual(reference_solution[i].z, tmpz[i])

    def test_cl_eval(self):
        """ Test the PyOpenCL implementation """

        if solver.HAS_CL:

            pa = self.pa
            func = self.ideal
            
            func.setup_cl(self.prog, self.ctx)

            func.cl_eval(self.q, self.ctx)

            pa.read_from_buffer()

            reference_solution = self.get_reference_solution()

            for i in range(self.np):
                self.assertAlmostEqual(reference_solution[i].x, pa.tmpx[i], 6)
                self.assertAlmostEqual(reference_solution[i].y, pa.tmpy[i], 6)
                self.assertAlmostEqual(reference_solution[i].z, pa.tmpz[i], 6)

if __name__ == '__main__':
    unittest.main()
