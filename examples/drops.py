""" A simple example in which two drops collide """

import pysph.solver.api as solver
import pysph.base.api as base
import pysph.sph.api as sph

import numpy

def get_circular_patch(name="", type=0, dx=0.05):
    
    x,y = numpy.mgrid[-1.05:1.05+1e-4:dx, -1.05:1.05+1e-4:dx]
    x = x.ravel()
    y = y.ravel()
 
    m = numpy.ones_like(x)*dx*dx
    h = numpy.ones_like(x)*2*dx
    rho = numpy.ones_like(x)

    p = 0.5*1.0*100*100*(1 - (x**2 + y**2))

    cs = numpy.ones_like(x) * 100.0

    u = 0*x
    v = 0*y

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

    return pa

def get_particles():

    f1 = get_circular_patch("fluid1")

    xlow, xhigh = min(f1.x), max(f1.x)

    f1.x += 1.2*(xhigh - xlow)
    f1.u[:] = -1.0

    f2 = get_circular_patch("fluid2")
    f2.u[:] = +1.0

    print "Number of particles: ", f1.get_number_of_particles() * 2.0

    return [f1,f2]


app = solver.Application()
app.process_command_line()

particles = app.create_particles(variable_h=False, callable=get_particles)

kernel = base.CubicSplineKernel(dim=2)

s = solver.FluidSolver(dim=2,
                       integrator_type=solver.PredictorCorrectorIntegrator)

s.set_final_time(1.0)
s.set_time_step(1e-4)

app.set_solver(s)

app.run()
