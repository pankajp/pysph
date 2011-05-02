""" Tests for the energy force functions """

import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph

if solver.HAS_CL:
    import pyopencl as cl

import numpy
import unittest
from os import path

NSquareLocator = base.NeighborLocatorType.NSquareNeighborLocator

class EnergyFunctionsTestCase(unittest.TestCase):

    def runTest(self):
        pass

    def setUp(self):
        """ The setup consists of four particles placed at the
        vertices of a unit square.

        The function tested is

        ..math::

        \frac{DU_a}{Dt} = \frac{1}{2}\sum_{b=1}^{N}m_b\left[
        \left(\frac{p_a}{\rho_a^2} + \frac{p_b}{\rho_b^2}\right)\,(v_a -
        v_b)\right]\,\nabla_a \cdot W_{ab}

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

        env = sph.EnergyEquationNoVisc.withargs()
        ewv = sph.EnergyEquation.withargs(alpha=1.0, beta=1.0,
                                          gamma=1.4, eta=0.1)


        self.env = env.get_func(pa,pa)
        self.ewv = ewv.get_func(pa,pa)
        
        self.env.kernel = base.CubicSplineKernel(dim=2)
        self.env.nbr_locator = \
                             base.Particles.get_neighbor_particle_locator(pa,
                                                                          pa)

        self.ewv.kernel = base.CubicSplineKernel(dim=2)
        self.ewv.nbr_locator = \
                             base.Particles.get_neighbor_particle_locator(pa,
                                                                          pa)

        self.setup_cl()

    def setup_cl(self):
        pass

class EnergyEquationNoViscTestCase(EnergyFunctionsTestCase):

    def setup_cl(self):
        pa = self.pa
        
        if solver.HAS_CL:
            self.ctx = ctx = cl.create_some_context()
            self.q = q = cl.CommandQueue(ctx)

            pa.setup_cl(ctx, q)
            
            pysph_root = solver.get_pysph_root()
            
            template = solver.cl_read(
                path.join(pysph_root, "sph/funcs/energy_funcs.clt"),
                function_name=self.env.cl_kernel_function_name,
                precision=self.precision)

            prog_src = solver.create_program(template, self.env)

            self.prog=cl.Program(ctx, prog_src).build(solver.get_cl_include())

    def get_reference_solution(self):
        """ Evaluate the force on each particle manually """
        
        pa = self.pa
        forces = []

        x,y,z,p,m,h,rho = pa.get('x','y','z','p','m','h','rho')
        u,v,w = pa.get('u','v','w')

        kernel = base.CubicSplineKernel(dim=2)

        for i in range(self.np):

            force = base.Point()
            xa, ya, za = x[i], y[i], z[i]
            ua, va, wa = u[i], v[i], w[i]

            ra = base.Point(xa,ya,za)
            Va = base.Point(ua,va,wa) 

            Pa, rhoa = p[i], rho[i]
            ha = h[i]

            for j in range(self.np):

                grad = base.Point()
                xb, yb, zb = x[j], y[j], z[j]
                ub, vb, wb = u[j], v[j], w[j]

                Pb, rhob = p[j], rho[j]
                hb, mb = m[j], h[j]

                havg = 0.5 * (ha + hb)

                rb = base.Point(xb, yb, zb)
                Vb = base.Point(ub, vb, wb)
        
                tmp = 0.5*mb * ( Pa/(rhoa*rhoa) + Pb/(rhob*rhob) )
                kernel.py_gradient(ra, rb, havg, grad)

                force.x += tmp * grad.dot(Va-Vb)

            forces.append(force)

        return forces

    def test_eval(self):
        """ Test the PySPH solution """

        pa = self.pa
        func = self.env

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
            func = self.env
            
            k = base.CubicSplineKernel(dim=2)

            func.setup_cl(self.prog, self.ctx)

            func.cl_eval(self.q, self.ctx, k)

            pa.read_from_buffer()

            reference_solution = self.get_reference_solution()

            for i in range(self.np):
                self.assertAlmostEqual(reference_solution[i].x, pa.tmpx[i], 6)
                self.assertAlmostEqual(reference_solution[i].y, pa.tmpy[i], 6)
                self.assertAlmostEqual(reference_solution[i].z, pa.tmpz[i], 6)

class EnergyEquationTestCase(EnergyFunctionsTestCase):

    def setup_cl(self):
        pa = self.pa
        
        if solver.HAS_CL:
            self.ctx = ctx = cl.create_some_context()
            self.q = q = cl.CommandQueue(ctx)

            pa.setup_cl(ctx, q)
            
            pysph_root = solver.get_pysph_root()
            
            template = solver.cl_read(
                path.join(pysph_root, "sph/funcs/energy_funcs.clt"),
                function_name=self.ewv.cl_kernel_function_name,
                precision=self.precision)

            self.ewv.set_cl_kernel_args()
            prog_src = solver.create_program(template, self.ewv)

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
            xa, ya, za = x[i], y[i], z[i]
            ua, va, wa = u[i], v[i], w[i]

            ra = base.Point(xa,ya,za)
            Va = base.Point(ua,va,wa)

            Pa, rhoa = p[i], rho[i]
            ha = h[i]

            for j in range(self.np):

                grad = base.Point()
                xb, yb, zb = x[j], y[j], z[j]
                Pb, rhob = p[j], rho[j]
                hb, mb = h[j], m[j]

                ub, vb, wb = u[j], v[j], w[j]
                Vb = base.Point(ub,vb,wb)

                havg = 0.5 * (hb + ha)

                rb = base.Point(xb, yb, zb)
        
                tmp = Pa/(rhoa*rhoa) + Pb/(rhob*rhob)
                kernel.py_gradient(ra, rb, havg, grad)

                vab = Va-Vb
                rab = ra-rb

                dot = vab.dot(rab)
                piab = 0.0

                if dot < 0.0:
                    alpha = 1.0
                    beta = 1.0
                    gamma = 1.4
                    eta = 0.1

                    cab = 0.5 * (cs[i] + cs[j])

                    rhoab = 0.5 * (rhoa + rhob)
                    muab = havg * dot

                    muab /= ( rab.norm() + eta*eta*havg*havg )

                    piab = -alpha*cab*muab + beta*muab*muab
                    piab /= rhoab

                tmp += piab
                tmp *= 0.5*mb

                force.x += tmp * ( vab.dot(grad) )

            forces.append(force)

        return forces

    def test_eval(self):
        """ Test the PySPH solution """

        pa = self.pa
        func = self.ewv

        k = base.CubicSplineKernel(dim=2)

        tmpx = pa.properties['tmpx']
        tmpy = pa.properties['tmpy']
        tmpz = pa.properties['tmpz']        

        func.eval(k, tmpx, tmpy, tmpz)

        reference_solution = self.get_reference_solution()

        for i in range(self.np):
            self.assertAlmostEqual(reference_solution[i].x, tmpx[i], 6)
            self.assertAlmostEqual(reference_solution[i].y, tmpy[i], 6)
            self.assertAlmostEqual(reference_solution[i].z, tmpz[i], 6)

    def test_cl_eval(self):
        """ Test the PyOpenCL implementation """

        if solver.HAS_CL:

            pa = self.pa
            func = self.ewv
            
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
