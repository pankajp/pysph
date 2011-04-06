
""" An example solver for the circular patch of fluid """

import numpy

import pysph.base.api as base

import pysph.sph.api as sph

from solver import Solver
from sph_equation import SPHOperation, SPHIntegration

Fluids = base.ParticleType.Fluid
Solids = base.ParticleType.Solid

def get_circular_patch(name="", type=0, dx=0.025):
    
    x,y = numpy.mgrid[-1.05:1.05+1e-4:dx, -1.05:1.05+1e-4:dx]
    x = x.ravel()
    y = y.ravel()
 
    m = numpy.ones_like(x)*dx*dx
    h = numpy.ones_like(x)*2*dx
    rho = numpy.ones_like(x)

    p = 0.5*1.0*100*100*(1 - (x**2 + y**2))

    cs = numpy.ones_like(x) * 100.0

    u = -100*x
    v = 100*y

    indices = []

    for i in range(len(x)):
        if numpy.sqrt(x[i]*x[i] + y[i]*y[i]) - 1 > 1e-10:
            indices.append(i)
            
    pa = base.get_particle_array(x=x, y=y, m=m, rho=rho, h=h, p=p, u=u, v=v,
                                 cs=cs,name=name, type=type)

    la = base.LongArray(len(indices))
    la.set_data(numpy.array(indices))

    pa.remove_particles(la)

    pa.set(idx=numpy.arange(len(pa.x)))
 
    print 'Number of particles: ', len(pa.x)
    
    return pa

class FluidSolver(Solver):

    def setup_solver(self):
        
        #create the sph operation objects

        self.add_operation(SPHOperation(

            sph.TaitEquation.withargs(co=100.0, ro=1.0),
            on_types=[Fluids],
            updates=['p', 'cs'],
            id='eos')
                           )

        self.add_operation(SPHIntegration(
            sph.SPHDensityRate.withargs(hks=False),
            from_types=[Fluids], on_types=[Fluids],
            updates=['rho'],
            id='density')
                           )
                                           

        self.add_operation(SPHIntegration(
            sph.MomentumEquation.withargs(alpha=0.01, beta=0.0, hks=False),
            from_types=[Fluids], on_types=[Fluids], 
            updates=['u','v'], 
            id='mom')
                           )

        self.add_operation_step([Fluids])
        self.add_operation_xsph(eps=0.1, hks=False)

#############################################################################
