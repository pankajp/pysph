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

        u = numpy.array([1, 0, 0, -1], numpy.float64)
        p = numpy.array([0, 0, 1, 1], numpy.float64)
        
        tmpx = numpy.zeros_like(x)
        tmpy = numpy.zeros_like(x)
        tmpz = numpy.zeros_like(x)

        self.pa = pa = base.get_particle_array(name="test", x=x,  y=y, z=z,
                                               m=m, u=u, p=p,
                                               tmpx=tmpx, tmpy=tmpy, tmpz=tmpz,
                                               cl_precision=self.precision)

        grad_func = sph.SPHPressureGradient.withargs()
        mom_func = sph.MomentumEquation.withargs(alpha=1.0, beta=1.0,
                                                 gamma=1.4, eta=0.1)


        self.grad_func = grad_func.get_func(pa,pa)
        self.mom_func = mom_func.get_func(pa,pa)
        
        self.grad_func.kernel = base.CubicSplineKernel(dim=2)
        self.grad_func.nbr_locator = \
                              base.Particles.get_neighbor_particle_locator(pa,
                                                                           pa)

        self.mom_func.kernel = base.CubicSplineKernel(dim=2)
        self.mom_func.nbr_locator = \
                             base.Particles.get_neighbor_particle_locator(pa,
                                                                          pa)

        self.setup_cl()

    def setup_cl(self):
        pass

class SPHPressureGradientTestCase(PressureForceTestCase):

    def setup_cl(self):
        pa = self.pa
        
        if solver.HAS_CL:
            self.ctx = ctx = cl.create_some_context()
            self.q = q = cl.CommandQueue(ctx)

            pa.setup_cl(ctx, q)
            
            pysph_root = solver.get_pysph_root()
            
            template = solver.cl_read(
                path.join(pysph_root, "sph/funcs/pressure_funcs.clt"),
                function_name=self.grad_func.cl_kernel_function_name,
                precision=self.precision)

            prog_src = solver.create_program(template, self.grad_func)

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
        func = self.grad_func

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
            func = self.grad_func
            
            func.setup_cl(self.prog, self.ctx)

            func.cl_eval(self.q, self.ctx)

            pa.read_from_buffer()

            reference_solution = self.get_reference_solution()

            for i in range(self.np):
                self.assertAlmostEqual(reference_solution[i].x, pa.tmpx[i], 6)
                self.assertAlmostEqual(reference_solution[i].y, pa.tmpy[i], 6)
                self.assertAlmostEqual(reference_solution[i].z, pa.tmpz[i], 6)

class MomentumEquationTestCase(PressureForceTestCase):

    def setup_cl(self):
        pa = self.pa
        
        if solver.HAS_CL:
            self.ctx = ctx = cl.create_some_context()
            self.q = q = cl.CommandQueue(ctx)

            pa.setup_cl(ctx, q)
            
            pysph_root = solver.get_pysph_root()
            
            template = solver.cl_read(
                path.join(pysph_root, "sph/funcs/pressure_funcs.clt"),
                function_name=self.mom_func.cl_kernel_function_name,
                precision=self.precision)

            self.mom_func.set_cl_kernel_args()
            prog_src = solver.create_program(template, self.mom_func)

            self.prog=cl.Program(ctx, prog_src).build(solver.get_cl_include())

    def get_reference_solution(self):
        """ Evaluate the force on each particle manually """
        
        pa = self.pa
        forces = []

        x,y,z,p,m,h,rho = pa.get('x','y','z','p','m','h','rho')
        u,v,w,cs = pa.get('u','v','w','cs')

        kernel = base.CubicSplineKernel(dim=2)

        for i in range(self.np):

            force = base.Point()
            xi, yi, zi = x[i], y[i], z[i]
            ui, vi, wi = u[i], v[i], w[i]

            ri = base.Point(xi,yi,zi)
            Va = base.Point(ui,vi,wi)

            Pi, rhoi = p[i], rho[i]
            hi = h[i]

            for j in range(self.np):

                grad = base.Point()
                xj, yj, zj = x[j], y[j], z[j]
                Pj, rhoj = p[j], rho[j]
                hj, mj = h[j], m[j]

                uj, vj, wj = u[j], v[j], w[j]
                Vb = base.Point(uj,vj,wj)

                havg = 0.5 * (hi + hj)

                rj = base.Point(xj, yj, zj)
        
                tmp = Pi/(rhoi*rhoi) + Pj/(rhoj*rhoj)
                kernel.py_gradient(ri, rj, havg, grad)

                vab = Va-Vb
                rab = ri-rj

                dot = vab.dot(rab)
                piab = 0.0

                if dot < 0.0:
                    alpha = 1.0
                    beta = 1.0
                    gamma = 1.4
                    eta = 0.1

                    cab = 0.5 * (cs[i] + cs[j])

                    rhoab = 0.5 * (rhoi + rhoj)
                    muab = havg * dot

                    muab /= ( rab.norm() + eta*eta*havg*havg )

                    piab = -alpha*cab*muab + beta*muab*muab
                    piab /= rhoab

                tmp += piab
                tmp *= -mj
                    
                force.x += tmp*grad.x
                force.y += tmp*grad.y
                force.z += tmp*grad.z

            forces.append(force)

        return forces

    def test_eval(self):
        """ Test the PySPH solution """

        pa = self.pa
        func = self.mom_func

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
            func = self.mom_func
            
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
