""" 2D Dam Break Over a dry bed. The case is described in "State of
the art classical SPH for free surface flows", Benedict D Rogers,
Robert A, Dalrymple and Alex J.C Crespo, Journal of Hydraulic
Research, Vol 48, Extra Issue (2010), pp 6-27


Setup:
------



x                   x !
x                   x !
x                   x !
x                   x !
x  o   o   o        x !
x    o   o          x !3m
x  o   o   o        x !
x    o   o          x !
x  o   o   o        x !
x                   x !
xxxxxxxxxxxxxxxxxxxxx |        o -- Fluid Particles
                               x -- Solid Particles
     -dx-                      dx = dy
_________4m___________

Y
|
|
|
|
|
|      /Z
|     /
|    /
|   /
|  /
| /
|/_________________X

Fluid particles are placed on a staggered grid. The nodes of the grid
are located at R = l*dx i + m * dy j with a two point bias (0,0) and
(dx/2, dy/2) refered to the corner defined by R. l and m are integers
and i and j are the unit vectors alon `X` and `Y` respectively.

For the Monaghan Type Repulsive boundary condition, a single row of
boundary particles is used with a boundary spacing delp = dx = dy.

For the Dynamic Boundary Conditions, a staggered grid arrangement is
used for the boundary particles.

Numerical Parameters:
---------------------

dx = dy = 0.012m
h = 0.0156 => h/dx = 1.3

Height of Water collumn = 2m
Length of Water collumn = 1m

Number of particles = 27639 + 1669 = 29308


ro = 1000.0
co = 10*sqrt(2*9.81*2) ~ 65.0
gamma = 7.0

Artificial Viscosity:
alpha = 0.5

XSPH Correction:
eps = 0.5

 """

import numpy
import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph

Fluid = base.ParticleType.Fluid
Solid = base.ParticleType.Solid

dx = dy = 0.012
h = 1.3*dx
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
    
    x, y =  numpy.mgrid[start_point.x:end_point.x+1e-10:spacing,
                        start_point.y:end_point.y+1e-10:spacing]

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
    

    #the left wall
    
    yb1 = get_1D_grid(0, container_height, dy)
    xb1 = numpy.zeros_like(yb1)
    nb1 = len(xb1)

    #bottom

    xb2 = get_1D_grid(dx, container_width - dx, dx)
    yb2 = numpy.zeros_like(xb2)
    nb2 = len(xb2)

    #right wall
    
    xb3 = xb1 + container_width
    yb3 = yb1.copy()
    nb3 = len(xb3)

    #staggered portion of the left wall
    yb4 = get_1D_grid(-dy/2, container_height-dy/2, dy)
    xb4 = numpy.ones_like(yb4) * -dx/2
    nb4 = len(xb4)

    # staggered portion for the bottom wall
    xb5 = get_1D_grid(dx/2, container_width-dx/2, dx)
    yb5 = numpy.ones_like(xb5) * -dy/2
    nb5 = len(xb5)

    #staggered portion for the right wall
    xb6 = xb4 + (container_width + dx)
    yb6 = yb4.copy()
    nb6 = len(xb6)

    nb = nb1 + nb2 + nb3 + nb4 + nb5 + nb6

    print "Number of Boundary Particles: ", nb
    
    xb = numpy.zeros(nb, float)
    yb = numpy.zeros(nb, float)

    idx = 0

    xb[:nb1] = xb1; yb[:nb1] = yb1

    idx += nb1

    xb[idx:idx+nb2] = xb2; yb[idx:idx+nb2] = yb2

    idx += nb2

    xb[idx:idx+nb3] = xb3; yb[idx:idx+nb3] = yb3
    idx += nb3

    xb[idx:idx+nb4] = xb4; yb[idx:idx+nb4] = yb4
    idx += nb4

    xb[idx:idx+nb5] = xb5; yb[idx:idx+nb5] = yb5
    idx += nb5

    xb[idx:] = xb6; yb[idx:] = yb6

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
    mf = numpy.ones_like(x) * dx * dy * ro
    rhof = numpy.ones_like(x) * ro
    csf = numpy.ones_like(x) * co
    
    fluid = base.get_particle_array(name="dam", type=Fluid,
                                    x=x, y=y, h=hf, m=mf, rho=rhof, cs=csf)

    return fluid


app = solver.Application()
app.process_command_line()


boundary = get_boundary_particles()
fluid = get_fluid_particles()

particles = base.Particles(arrays=[fluid, boundary])
app.particles = particles
s = solver.Solver(base.HarmonicKernel(dim=2, n=3), solver.EulerIntegrator)

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

s.set_final_time(0.4)
s.set_time_step(1e-5)

app.set_solver(s)

app.run()
