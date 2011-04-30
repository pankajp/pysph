""" Tests for the pressure force functions """

import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph

if solver.HAS_CL:
    import pyopencl as cl

import numpy
import unittest
from os import path

NSquareLocator = base.NeighborLocatorType.NSquareNeighborLocator

class PressureForceTestCase(unittest.TestCase):

    def runTest(self):
        pass

    def setUp(self):
        """ The setup consists of four particles placed at the
        vertices of a unit square. The pressure gradient term to be
        tested is

        ..math::

                \frac{\nablaP}{\rho}_i = \sum_{j=1}^{4}
                -m_j(\frac{Pa}{\rho_a^2} + \frac{Pb}{\rho_b^2})\nabla W_{ab}

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

        func = sph.SPHPressureGradient.withargs()
        self.func = func = func.get_func(pa,pa)

        self.func.kernel = base.CubicSplineKernel(dim=2)
        func.nbr_locator = base.Particles.get_neighbor_particle_locator(pa,pa)

        if solver.HAS_CL:
            self.ctx = ctx = cl.create_some_context()
            self.q = q = cl.CommandQueue(ctx)

            pa.setup_cl(ctx, q)
            
            pysph_root = solver.get_pysph_root()
            
            template = solver.cl_read(
                path.join(pysph_root, "sph/funcs/pressure_funcs.clt"),
                function_name=self.func.cl_kernel_function_name,
                precision=self.precision)

            prog_src = solver.create_program(template, self.func)

            self.prog=cl.Program(ctx, prog_src).build(solver.get_cl_include())

    def get_reference_solution(self):
        """ Evaluate the force on each particle manually """
        
        pa = self.pa
        forces = []

        x,y,z,p,m,h,rho = pa.get('x','y','z','p','m','h','rho')

        kernel = base.CubicSplineKernel(dim=2)

        for i in range(self.np):

            force = base.Point()
            xi, yi, zi = x[i], y[i], z[i]

            ri = base.Point(xi,yi,zi)

            Pi, rhoi = p[i], rho[i]
            hi = h[i]

            for j in range(self.np):

                grad = base.Point()
                xj, yj, zj = x[j], y[j], z[j]
                Pj, rhoj = p[j], rho[j]
                hj, mj = m[j], h[j]

                havg = 0.5 * (hi + hj)

                rj = base.Point(xj, yj, zj)
        
                tmp = -mj * ( Pi/(rhoi*rhoi) + Pj/(rhoj*rhoj) )
                kernel.py_gradient(ri, rj, havg, grad)

                force.x += tmp*grad.x
                force.y += tmp*grad.y
                force.z += tmp*grad.z

            forces.append(force)

        return forces

    def test_eval(self):
        """ Test the PySPH solution """

        pa = self.pa
        func = self.func

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
            func = self.func
            
            k = base.CubicSplineKernel(dim=2)

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
