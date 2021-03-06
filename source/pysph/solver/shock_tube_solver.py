""" A simple shock tube solver """

import numpy

import pysph.base.api as base
import pysph.sph.api as sph

from solver import Solver
from sph_equation import SPHOperation, SPHIntegration

Fluids = base.ParticleType.Fluid
Solids = base.ParticleType.Solid

def standard_shock_tube_data(name="", type=0):
    """ Standard 400 particles shock tube problem """
    
    dxl = 0.001875
    dxr = dxl*4
    
    x = numpy.ones(400, float)
    x[:320] = numpy.arange(-0.6, -dxl+1e-4, dxl)
    x[320:] = numpy.arange(dxr, 0.6+1e-4, dxr)

    m = numpy.ones_like(x)*dxl
    h = numpy.ones_like(x)*2*dxr

    rho = numpy.ones_like(x)
    rho[320:] = 0.25
    
    u = numpy.zeros_like(x)
    
    e = numpy.ones_like(x)
    e[:320] = 2.5
    e[320:] = 1.795

    p = 0.4*rho*e

    cs = numpy.sqrt(1.4*p/rho)

    idx = numpy.arange(400)
    
    return base.get_particle_array(name=name,x=x,m=m,h=h,rho=rho,p=p,e=e,
                                   cs=cs,type=type, idx=idx)

class ShockTubeSolver(Solver):
    
    def setup_solver(self):

        kernel = base.CubicSplineKernel(dim=1)

        #create the sph operation objects

        self.add_operation(SPHOperation(

            sph.SPHRho.withargs(),
            from_types=[Fluids], on_types=[Fluids], 
            updates=['rho'], id = 'density', kernel=kernel)

                           )

        self.add_operation(SPHOperation(

            sph.IdealGasEquation.withargs(),
            on_types = [Fluids], updates=['p', 'cs'], id='eos',
            kernel=kernel)

                           )

        self.add_operation(SPHIntegration(

            sph.MomentumEquation.withargs(),
            from_types=[Fluids], on_types=[Fluids], 
            updates=['u'], id='mom', kernel=kernel)

                           )
        
        self.add_operation(SPHIntegration(

            sph.EnergyEquation.withargs(hks=False),
            from_types=[Fluids],
            on_types=[Fluids], updates=['e'], id='enr',
            kernel=kernel)

                           )

        # Indicate that stepping is only needed for Fluids

        self.add_operation_step([Fluids])


#############################################################################
    
        
        
