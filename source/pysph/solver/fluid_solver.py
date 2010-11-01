
""" An example solver for the circular patch of fluid """

import numpy

import pysph.base.api as base

import pysph.sph.api as sph

from solver import Solver
from sph_equation import SPHSummation, SPHAssignment, SPHSummationODE, \
    SPHSimpleODE

Fluids = base.ParticleType.Fluid
Solids = base.ParticleType.Solid

def get_circular_patch(name="", type=0):
    
    x,y = numpy.mgrid[-1.05:1.05+1e-4:0.025, -1.05:1.05+1e-4:0.025]
    x = x.ravel()
    y = y.ravel()
 
    m = numpy.ones_like(x)*0.025*0.025
    h = numpy.ones_like(x)*0.05
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
 
    print 'Number of particles: ', len(pa.x)
    
    return pa


class FluidSolver(Solver):

    def setup_solver(self):
        
        #create the sph operation objects

        momentum = sph.MomentumEquation(alpha=0.01, beta=0.0)        

        self.add_operation(SPHAssignment(sph.TaitEquation(co=100.0, ro=1.0),
                                         on_types=[Fluids],
                                         updates=['p', 'cs'],
                                         id='eos')
                           )

        self.add_operation(SPHSummationODE(sph.SPHDensityRate(),
                                           from_types=[Fluids],
                                           on_types=[Fluids],
                                           updates=['rho'],
                                           id='density')
                           )
                                           

        self.add_operation(SPHSummationODE(momentum, 
                                           from_types=[Fluids],
                                           on_types=[Fluids], 
                                           updates=['u','v'], 
                                           id='mom')
                           )

        self.add_operation(SPHSummationODE(sph.XSPHCorrection(eps=0.1),
                                           from_types=[Fluids],
                                           on_types=[Fluids],
                                           updates=['x','y'],
                                           id='xsph')
                           )


        self.add_operation(SPHSimpleODE(sph.PositionStepping(),
                                        on_types=[Fluids], 
                                        updates=['x','y'], id='step')
                           )

#############################################################################
