# cython: profile=True
"""module to test the timings for 1 iteration of dam_break problem """

import time

import numpy
import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph

Fluid = base.ParticleType.Fluid
Solid = base.ParticleType.Solid

#h = 0.0156
h = 0.0390
#h = 0.01
dx = dy = h/1.3
ro = 1000.0
co = 65.0
gamma = 7.0
alpha = 0.5
eps = 0.5

fluid_collumn_height = 2.0
fluid_collumn_width  = 1.0
container_height = 3.0
container_width  = 4.0

B = co*co*ro/gamma

def get_1D_grid(start, end, spacing):
    """ Return an array of points in 1D

    Parameters:
    -----------
    start -- the starting coordinate value
    end -- the ending coordinate value
    spacing -- the uniform spacing between the points

    Notes:
    ------
    Uses numpy arange to get the points!
    
    """

    return numpy.arange(start, end+1e-10, spacing)

def get_2D_grid(start_point, end_point, spacing):
    """ Return a 2D array of points by calling numpy's mgrid

    Parameters:
    -----------
    start_point -- the starting corner point for the rectangle
    end_point -- the ending corner point for the rectangle
    spacing -- uniform spacing in x and y

    """
    
    x, y =  numpy.mgrid[start_point.x:end_point.x:spacing,
                        start_point.y:end_point.y:spacing]

    x = x.ravel(); y = y.ravel()

    return x, y

def get_2D_staggered_grid(bias_point_1, bias_point_2, end_point, spacing):
    """ Return a staggered cartesian grid in 2D

    Parameters:
    -----------
    bias_point_1 -- the first grid starting point
    bias_point_2 -- the second grid starting point
    end_point -- the maximum `x` and `y` for the grid
    spacing -- uniform spacing in `x` and `y`

    """

    x1, y1 = get_2D_grid(bias_point_1, end_point, spacing)
    x2, y2 = get_2D_grid(bias_point_2, end_point, spacing)
    
    x = numpy.zeros(len(x1)+len(x2), float)
    y = numpy.zeros(len(x1)+len(x2), float)

    x[:len(x1)] = x1; y[:len(x1)] = y1
    x[len(x1):] = x2; y[len(x1):] = y2

    return x, y

def get_boundary_particles():
    """ Get the particles corresponding to the dam and fluids """
    
    #left wall
    ylw = get_1D_grid(0, container_height, dy)
    xlw = numpy.zeros_like(ylw)
    nb1 = len(ylw)

    #bottom
    xbs = get_1D_grid(dx, container_width+dx, dx)
    ybs = numpy.zeros_like(xbs)
    nb3 = len(xbs)

    max_xb = numpy.max(xbs)
    
    #staggered left wall
    yslw = get_1D_grid(-dx/2, container_height, dx)
    xslw = numpy.ones_like(yslw) * -dx/2
    nb4 = len(yslw)

    #staggered bottom
    xsb = get_1D_grid(dx/2, container_width+dx+dx, dx)
    ysb = numpy.ones_like(xsb) * -dy/2
    nb6 = len(xsb)

    max_xsb = numpy.max(xsb)

    #right wall
    yrw = numpy.arange(dx, container_height, dx)
    xrw = numpy.ones_like(yrw) * max_xb
    nb2 = len(yrw)

    #staggered right wall
    ysrw = numpy.arange(dy/2, container_height, dy)
    xsrw = numpy.ones_like(ysrw) * max_xsb
    nb5 = len(ysrw)

    nb = nb1 + nb2 + nb3 + nb4 + nb5 + nb6

    print "Number of Boundary Particles: ", nb
    
    xb = numpy.zeros(nb, float)
    yb = numpy.zeros(nb, float)

    idx = 0

    xb[:nb1] = xlw; yb[:nb1] = ylw

    idx += nb1

    xb[idx:idx+nb2] = xrw; yb[idx:idx+nb2] = yrw

    idx += nb2

    xb[idx:idx+nb3] = xbs; yb[idx:idx+nb3] = ybs
    idx += nb3

    xb[idx:idx+nb4] = xslw; yb[idx:idx+nb4] = yslw
    idx += nb4

    xb[idx:idx+nb5] = xsrw; yb[idx:idx+nb5] = ysrw
    idx += nb5

    xb[idx:] = xsb; yb[idx:] = ysb

    hb = numpy.ones_like(xb)*h
    mb = numpy.ones_like(xb)*dx*dy*ro
    rhob = numpy.ones_like(xb) * ro

    cb = numpy.ones_like(xb)*co

    boundary = base.get_particle_array(name="boundary", type=Solid, 
                                       x=xb, y=yb, h=hb, rho=rhob, cs=cb,
                                       m=mb)

    return boundary

def get_fluid_particles():
    
    x, y = get_2D_staggered_grid(base.Point(dx, dx), base.Point(dx/2, dx/2), 
                                 base.Point(1.0,2.0), dx)

    print 'Number of fluid particles: ', len(x)

    hf = numpy.ones_like(x) * h
    mf = numpy.ones_like(x) * dx * dy * ro * 0.5
    rhof = numpy.ones_like(x) * ro
    csf = numpy.ones_like(x) * co
    
    fluid = base.get_particle_array(name="fluid", type=Fluid,
                                    x=x, y=y, h=hf, m=mf, rho=rhof, cs=csf)

    return fluid

def get_particles():
    fluid = get_fluid_particles()
    boundary = get_boundary_particles()

    return [fluid, boundary]


def dam_break_run():
    """ dam break problem runtime bench for 1 timestep """
    cdef double t, t2, tt
    cdef dict ret = {}
    get_time = time.time
    
    t = get_time()
    app = solver.Application()
    t = get_time() - t
    ret['create Application'] = t
    
    t = get_time()
    app.process_command_line()
    t = get_time() - t
    ret['process_command_line'] = t
    
    t = get_time()
    particles = app.create_particles(get_particles)
    t = get_time() - t
    ret['create_particles'] = t
    
    t = get_time()
    s = solver.Solver(base.CubicSplineKernel(dim=2), 
                      solver.RK2Integrator)
    t = get_time() - t
    ret['create Solver'] = t
    
    t = get_time()
    #Equation of state
    s.add_operation(solver.SPHAssignment(
            
            sph.TaitEquation(co=co, ro=ro), 
            on_types=[Fluid, Solid], 
            updates=['p', 'cs'],
            id='eos')
    
                    )
    
    #Continuity equation
    s.add_operation(solver.SPHSummationODE(
            
            sph.SPHDensityRate(), 
            on_types=[Fluid, Solid], from_types=[Fluid, Solid], 
            updates=['rho'], id='density')
                    
                    )
    
    #momentum equation
    s.add_operation(solver.SPHSummationODE(
            
            sph.MomentumEquation(alpha=alpha, beta=0.0),
            on_types=[Fluid], from_types=[Fluid, Solid],  
            updates=['u','v'], id='mom')
                        
                    )
    
    #Gravity force
    s.add_operation(solver.SPHSimpleODE(
            
            sph.GravityForce(gy=-9.81),
            on_types=[Fluid],
            updates=['u','v'],id='gravity')
                    
                    )
    
    #XSPH correction
    s.add_operation(solver.SPHSummationODE(
            
            sph.XSPHCorrection(eps=eps), 
            on_types=[Fluid], from_types=[Fluid],
            updates=['x','y'], id='xsph')
                    
                    )
    
    #Position stepping
    s.add_operation(solver.SPHSimpleODE(
            
            sph.PositionStepping(), 
            on_types=[Fluid], 
            updates=['x','y'], id='step')
                    
                    )
    t = get_time() - t
    ret['add_operations'] = t
    
    t = get_time()
    app.set_solver(s)
    t = get_time() - t
    ret['set_solver'] = t
    
    s.set_final_time(0.9e-4)
    s.set_time_step(1e-4)
    s.pfreq = 10
    
    t = get_time()
    app.run()
    t = get_time() - t
    ret['run 1 iteration'] = t
    
    t = get_time()
    s.dump_output(s.t)
    t = get_time() - t
    ret['dump_output'] = t
    
    return ret


cdef list funcs = [dam_break_run]


cpdef bench():
    """returns a list of a dict of dam_break problem timings"""
    cdef list timings = []
    for func in funcs:
        timings.append(func())
    return timings # dict of test:time
    
if __name__ == '__main__':
    print bench()
