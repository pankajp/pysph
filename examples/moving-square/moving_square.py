""" The moving square test case is part of the SPHERIC benchmark
tests. Refer to the document for the test details. 

Numerical Parameters:
---------------------

dx = dy = 0.005
h = 0.0065 => h/dx = 1.3

Length of Box = 10
Height of Box = 5

Number of particles = 27639 + 1669 = 29308

ro = 1000.0
Vmax = 1.0
co = 15 (15 * Vmax)
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
DummyFluid = base.ParticleType.DummyFluid

dx = 0.05
h = 1.3*dx
ro = 1000.0
co = 15.0
gamma = 7.0
alpha = 0.5
eps = 0.5

box_length = 10.0
box_height = 5.0
square_side = 1.0

B = co*co*ro/gamma
m = ro*dx*dx

pi = numpy.pi
pi2 = pi/2.0


class MoveSquare:

    def __init__(self, fname = "Motion_Body.dat"):
        self.original_position = 1.5

        motion = numpy.loadtxt(fname)

        self.time = motion[:,0]
        self.disp = motion[:,3]

    def eval(self, particles, count, time):

        square = particles.get_named_particle_array("square")
        x = square.get('x')

        new_pos = numpy.interp(time, self.time, self.disp)

        displacement = new_pos - self.original_position

        x += displacement

        square.set(x=x)

def get_wall():
    """ Get the wall particles """
    
    left = base.Line(base.Point(), box_height, pi2)
    
    top = base.Line(base.Point(0, box_height), box_length, 0)

    right = base.Line(base.Point(box_length, box_height), box_height, pi+pi2)

    bottom = base.Line(base.Point(box_length), box_length, pi)

    box_geom = base.Geometry('box', [left, top, right, bottom], is_closed=True)
    box_geom.mesh_geometry(dx)
    box = box_geom.get_particle_array(re_orient=False)

    box.m[:] = m
    box.h[:] = h

    return box

def get_square():
    """ Get the square particle array """
    
    left = base.Line(base.Point(1,2), square_side, pi2)
    
    top = base.Line(base.Point(1,3), square_side, 0)

    right = base.Line(base.Point(2,3), square_side, pi+pi2)

    bottom = base.Line(base.Point(2,2), square_side, pi)

    square_geom = base.Geometry('square', [left, top, right, bottom], 
                                is_closed=True)

    square_geom.mesh_geometry(dx)
    square = square_geom.get_particle_array(name="square", re_orient=True)

    square.m[:] = m
    square.h[:] = h

    return square

def get_fluid():
    """ Get the fluid particle array """

    x, y = numpy.mgrid[dx: box_length - 1e-10: dx,
                       dx: box_height - 1e-10: dx]
    
    xf, yf = x.ravel(), y.ravel()

    mf = numpy.ones_like(xf) * m
    hf = numpy.ones_like(xf) * h
    
    rhof = numpy.ones_like(xf) * ro
    cf = numpy.ones_like(xf) * co
    pf = numpy.zeros_like(xf)
    
    fluid = base.get_particle_array(name="fluid", type=Fluid, 
                                    x=xf, y=yf, h=hf, rho=rhof, c=cf, p=pf)

    # remove indices within the square

    indices = []

    np = fluid.get_number_of_particles()
    x, y  = fluid.get('x','y')

    for i in range(np):
        if 1.0 -dx/2 <= x[i] <= 2.0 + dx/2:
            if 2.0 - dx/2 <= y[i] <= 3.0 + dx/2:
                indices.append(i)
                
    to_remove = base.LongArray(len(indices))
    to_remove.set_data(numpy.array(indices))

    fluid.remove_particles(to_remove)

    return fluid

def get_dummy_particles():
    x, y = numpy.mgrid[-5*dx: box_length + 5*dx + 1e-10: dx,
                       -5*dx: box_height + 5*dx + 1e-10: dx]

    xd, yd = x.ravel(), y.ravel()

    md = numpy.ones_like(xd) * m
    hd = numpy.ones_like(xd) * h
    
    rhod = numpy.ones_like(xd) * ro
    cd = numpy.ones_like(xd) * co
    pd = numpy.zeros_like(xd)
    
    dummy_fluid = base.get_particle_array(name="dummy_fluid",
                                          type=Fluid, x=xd, y=yd,
                                          h=hd, rho=rhod, c=cd, p=pd)

    # remove indices within the square

    indices = []

    np = dummy_fluid.get_number_of_particles()
    x, y  = dummy_fluid.get('x','y')

    for i in range(np):
        if -dx/2 <= x[i] <= box_length + dx/2:
            if - dx/2 <= y[i] <= box_height+ dx/2:
                indices.append(i)
                
    to_remove = base.LongArray(len(indices))
    to_remove.set_data(numpy.array(indices))

    dummy_fluid.remove_particles(to_remove)

    return dummy_fluid

def get_particles():
    wall = get_wall()
    square = get_square()
    fluid = get_fluid()
    dummy_fluid = get_dummy_particles()

    return [wall, square, fluid, dummy_fluid]

app = solver.Application()
app.process_command_line()

particles = app.create_particles(False, get_particles)

s = solver.Solver(base.CubicSplineKernel(dim=2), 
                  solver.PredictorCorrectorIntegrator)

# Equation of state

s.add_operation(solver.SPHAssignment(
        
        sph.TaitEquation(co=co, ro=ro), 
        on_types=[Fluid], 
        updates=['p', 'cs'],
        id='eos')

                )

# Continuity equation

s.add_operation(solver.SPHSummationODE(
        
        sph.SPHDensityRate(), 
        on_types=[Fluid], from_types=[Fluid, DummyFluid], 
        updates=['rho'], id='density')
                
                )

# momentum equation

s.add_operation(solver.SPHSummationODE(
        
        sph.MomentumEquation(alpha=alpha, beta=0.0),
        on_types=[Fluid], from_types=[Fluid, DummyFluid],
        updates=['u','v'], id='mom')
                    
                )

# monaghan boundary force

s.add_operation(solver.SPHSummationODE(
        
        sph.MonaghanBoundaryForce(delp=dx),
        on_types=[Fluid], from_types=[Solid], updates=['u','v'],
        id='bforce')
                
                )

# Position stepping and XSPH correction

s.to_step([Fluid])
s.set_xsph(eps=eps)

# add post step and pre step functions for movement

s.set_final_time(3.0)
s.set_time_step(1e-5)

s.post_step_functions.append(MoveSquare())

app.set_solver(s)

app.run()
