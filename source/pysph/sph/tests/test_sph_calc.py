"""
Tests for the sph_calc module.
"""
# standard imports
import unittest
import numpy

import pysph.solver.api as solver
import pysph.base.api as base
import pysph.sph.api as sph

Fluid = base.ParticleType.Fluid
Solid = base.ParticleType.Solid

def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-16."""
    return numpy.allclose(x, y, atol=1e-16, rtol=0)

##############################################################################

def test_sph_calc():

    x = numpy.array([0,])
    y = numpy.array([0,])
    z = numpy.array([0,])
    h = numpy.ones_like(x)

    pa = base.get_particle_array(name="test", x=x, y=y, z=z,h=h)
    particles = base.Particles(arrays=[pa,])
    kernel = base.CubicSplineKernel(dim=1)

    vector_force1 = sph.VectorForce.withargs(force=base.Point(1,1,1))
    vector_force2 = sph.VectorForce.withargs(force=base.Point(1,1,1))

    func1 = vector_force1.get_func(pa,pa)
    func2 = vector_force2.get_func(pa,pa)

    calc = sph.SPHCalc(particles=particles, sources=[pa,pa], dest=pa,
                       kernel=kernel, funcs=[func1, func2],
                       updates=['u','v','w'], integrates=True)

    # evaluate the calc. Accelerations are stored in tmpx, tmpy and tmpz

    calc.sph('tmpx', 'tmpy', 'tmpz')

    tmpx, tmpy, tmpz = pa.get('tmpx', 'tmpy', 'tmpz')

    # the acceleration should be 2 in each direction

    assert ( abs(tmpx[0] - 2.0) < 1e-16 )
    assert ( abs(tmpy[0] - 2.0) < 1e-16 )
    assert ( abs(tmpz[0] - 2.0) < 1e-16 )
        
if __name__ == '__main__':
    test_sph_calc()
