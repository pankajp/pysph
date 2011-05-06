"""
Tests for the density_funcs module.
"""

# standard imports
import unittest
import numpy
import os

#import opencl if available
import pysph.solver.cl_utils as cl_utils
import pyopencl as cl

# local imports
import pysph.base.api as base
import pysph.sph.api as sph
import pysph.solver.api as solver

from pysph.sph.funcs.density_funcs import SPHRho, SPHDensityRate
from pysph.base.particle_array import ParticleArray
from pysph.base.kernels import Poly6Kernel, CubicSplineKernel

def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-16."""
    return numpy.allclose(x, y, atol=1e-16, rtol=0)

NSquareLocator = base.NeighborLocatorType.NSquareNeighborLocator

class DensityFunctionsTestCase(unittest.TestCase):

    def runTest(self):
        pass

    def setUp(self):
        """ The setup consists of four particles placed at the
        vertices of a unit square. 
        
        """
        
        self.precision = "single"

        self.np = 4

        x = numpy.array([0, 0, 1, 1], numpy.float64)
        y = numpy.array([0, 1, 1, 0], numpy.float64)

        z = numpy.zeros_like(x)
        m = numpy.ones_like(x)

        u = numpy.array([1, 0, 0, -1], numpy.float64)
        p = numpy.array([0, 0, 1, 1], numpy.float64)
        
        self.pa = pa = base.get_particle_array(name="test", x=x,  y=y, z=z,
                                               m=m, u=u, p=p,
                                               cl_precision=self.precision)

        self.particles = particles = base.Particles([pa,])

        sphrho_func = sph.SPHRho.withargs()
        density_rate_func = sph.SPHDensityRate.withargs()

        self.sphrho_func = sphrho_func.get_func(pa,pa)
        self.density_rate_func = density_rate_func.get_func(pa,pa)
        
        self.sphrho_func.kernel = base.CubicSplineKernel(dim=2)
        self.density_rate_func.kernel = base.CubicSplineKernel(dim=2)

        self.rho_calc = sph.SPHCalc(particles, [pa,], pa,
                                    base.CubicSplineKernel(dim=2),
                                    [self.sphrho_func,], updates=['rho']
                                    )

        self.rho_calc_cl = sph.CLCalc(particles, [pa,], pa,
                                      base.CubicSplineKernel(dim=2),
                                      [self.sphrho_func,], updates=['rho']
                                      )

        if solver.HAS_CL:
            self.ctx = ctx = cl.create_some_context()
            self.q = q = cl.CommandQueue(ctx)
            self.rho_calc_cl.setup_cl(ctx)

class SummationDensityTestCase(DensityFunctionsTestCase):

    def get_reference_solution(self):
        """ Evaluate the force on each particle manually """
        
        pa = self.pa
        rhos = []

        x,y,z,p,m,h,rho = pa.get('x','y','z','p','m','h','rho')

        kernel = base.CubicSplineKernel(dim=2)

        for i in range(self.np):

            rho = 0.0
            xi, yi, zi = x[i], y[i], z[i]

            ri = base.Point(xi,yi,zi)

            hi = h[i]

            for j in range(self.np):

                grad = base.Point()
                xj, yj, zj = x[j], y[j], z[j]
                hj, mj = m[j], h[j]

                havg = 0.5 * (hi + hj)

                rj = base.Point(xj, yj, zj)
        
                wij = kernel.py_function(ri, rj, havg)

                rho += mj*wij

            rhos.append(rho)

        return rhos

    def test_eval(self):
        """ Test the PySPH solution """

        pa = self.pa
        calc = self.rho_calc

        k = base.CubicSplineKernel(dim=2)

        calc.sph()
        tmpx = pa.properties['_tmpx']

        reference_solution = self.get_reference_solution()

        for i in range(self.np):
            self.assertAlmostEqual(reference_solution[i], tmpx[i])

    def test_cl_eval(self):
        """ Test the PyOpenCL implementation """

        if solver.HAS_CL:

            pa = self.pa
            calc = self.rho_calc_cl
            
            calc.sph()
            pa.read_from_buffer()

            reference_solution = self.get_reference_solution()

            for i in range(self.np):
                self.assertAlmostEqual(reference_solution[i], pa._tmpx[i], 6)

        
if __name__ == '__main__':
    unittest.main()
