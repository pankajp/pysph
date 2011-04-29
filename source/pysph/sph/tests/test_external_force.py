""" Tests for the External Force Functions """

import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph

if solver.HAS_CL:
    import pyopencl as cl

import numpy
import unittest
from os import path

class NBodyForceTestCase(unittest.TestCase):
    """ Simple test for the NBodyForce """

    def runTest(self):
        pass

    def setUp(self):
        """ The setup consists of four particles placed at the
        vertices of a unit square. The force function to be tested is:

        ..math::

                f_i = \sum_{j=1}^{4} \frac{m_j}{|x_j - x_i|^3 +
                \eps}(x_j - x_i)

        The mass of each particle is 1

        """
        
        self.precision = "single"

        self.np = 4

        x = numpy.array([0, 0, 1, 1], numpy.float64)
        y = numpy.array([0, 1, 1, 0], numpy.float64)
        z = numpy.zeros_like(x)
        m = numpy.ones_like(x)
        tmpx = numpy.zeros_like(x)
        tmpy = numpy.zeros_like(x)
        tmpz = numpy.zeros_like(x)

        self.pa = pa = base.get_particle_array(name="test", x=x,  y=y, z=z,
                                               m=m, tmpx=tmpx, tmpy=tmpy,
                                               tmpz=tmpz,
                                               cl_precision=self.precision)

        self.func = func = sph.NBodyForce.get_func(pa, pa)

        self.eps = func.eps

        if solver.HAS_CL:
            self.ctx = ctx = cl.create_some_context()
            self.q = q = cl.CommandQueue(ctx)

            pa.setup_cl(ctx, q)
            
            pysph_root = solver.get_pysph_root()
            
            template = solver.cl_read(
                path.join(pysph_root, "sph/funcs/external_force.cl"),
                function_name=func.cl_kernel_function_name,
                precision=self.precision)

            prog_src = solver.create_program(template, func)

            self.prog = cl.Program(ctx, prog_src).build(solver.get_cl_include())
        
    def get_reference_solution(self):
        """ Evaluate the force on each particle manually """

        pa = self.pa
        def get_force(i):
            xi = pa.x[i]; yi = pa.y[i]

            force = base.Point()

            for j in range(self.np):
                xj = pa.x[j]; yj = pa.y[j]

                xji = xj - xi; yji = yj - yi
                dist = numpy.sqrt( xji**2 + yji**2 )

                invr = 1.0/(dist + self.eps)
                invr3 = invr * invr * invr
              
                if not ( i == j ):
                    
                    force.x += invr3 * xji
                    force.y += invr3 * yji

            return force

        forces = [get_force(i) for i in range(self.np)]
        return forces

    def test_eval(self):
        """ Test the PySPH solution """

        pa = self.pa
        func = self.func

        k = base.CubicSplineKernel()

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
            func = self.func
            
            k = base.CubicSplineKernel()

            func.setup_cl(self.prog, self.ctx)

            func.cl_eval(self.q, self.ctx, k)

            pa.read_from_buffer()

            reference_solution = self.get_reference_solution()

            for i in range(self.np):
                self.assertAlmostEqual(reference_solution[i].x, pa.tmpx[i], 6)
                self.assertAlmostEqual(reference_solution[i].y, pa.tmpy[i], 6)
                self.assertAlmostEqual(reference_solution[i].z, pa.tmpz[i], 6)

if __name__ == '__main__':
    unittest.main()
