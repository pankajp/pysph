An Overview of PySPH
=======================
PySPH is a framework for Smoothed Particle Hydrodynamics (`SPH <http://en.wikipedia.org/wiki/Smoothed-particle_hydrodynamics>`_) 
in `Python <http://www.python.org>`_. The framework allows us to define arbitrary collections
of partcles and forces acting on them. The corresponding initial value problem is then integrated
in time to obtain the desired solution. The best way to understand this is to consider an example.

N-Body simulation using PySPH
-------------------------------
Consider a system of points which is governed by the following equations:
 .. math::

	\frac{D\vec{v_i}}{Dt} = \sum_{j=1}^{N} \frac{m_j}{|x_j - x_i|^3} (\vec{x_j} - \vec{x_i})\,\, \forall i \neq j
	\frac{Dx_i}{Dt} = \vec{v_i}

Thus, given the initial positions and velocities of the particles, we can numerically integrate the system to some 
final time. Let's see how to do this in PySPH!

..  sourcecode:: python
	:linenos:

	import pysph.base.api as base
	import pysph.solver.api as solver
	import pysph.sph.api as sph
	import numpy
	
	Fluid = base.ParticleType.Fluid
	
	# Generate random points in the cube [-1, 1] X [-1, 1] X [-1,1]
	x = numpy.random.ramdom(1000) * 2.0 - 1.0
	y = numpy.random.ramdom(1000) * 2.0 - 1.0
	z = numpy.random.random(1000) * 2.0 - 1.0
	m = numpy.random.random(1000)
	
	pa = base.get_particle_array(name="test", type=Fluid, x=x, y=y, z=z, m=m)
	particles = base.Particles(arrays=[pa,])
	
	kernel = base.CubicSplineKernel(dim=3)
	s = solver.Solver(kernel, solver.PredictorCorrectorIntegrator)
	
	s.add_operation(solver.SPHIntegration
	
			sph.NBodyForce.withargs(),
			on_types=[Fluid], from_types=[Fluid],
			updates=['u','v','w'], id='nbody_force'
			
			)
			
	s.to_step([Fluid,])
	s.set_final_time(1.0)
	s.set_time_step(1e-4)
	
	s.setup_integrator(particles)
	
	s.solve()

This example demonstrates the basic steps used to solve a particular problem with PySPH:

* Create the particles with desired properties.
* Construct a solver using an appropriate integrator.
* Add forces to the system indicating the sources (from_types) and targets (on_types)
* Run the solver.

A collection of particles of a particular type is represented by a **ParticleArray**.::

	pa = base.get_particle_array(name="test", type=Fluid, x=x ...)

Here we request an array of fluids with properties given by numpy arrays. See :doc:`particle_array` for
more information on the use of **ParticleArrays**. 

A collection of arrays is represented by **Particles** as::

	particles = base.Particles(arrays=[pa1,pa2,...])

We can construct an arbitrary number of arrays and pass it as a list to the **Particles**'s constructor.
Among other things, this class constructs the internal indexing scheme which is used for fast neighbor
queries (see :doc:`nnps`).

We then construct a solver with a kernel and state that we must use a predictor corrector scheme
for integration of the system::

	s = solver.Solver(kernel=kernel, integrator_type=solver.PredictorCorrectorIntegrator)

Now we need to tell PySPH how do the particles interact with each other. This is done via the following
paradigm. We add an operation to the solver. The operation acts on certain types of particles, using 
other types of particles for the origin of the interaction. The operation is meant to update certain 
variables on the target particles::

	s.add_operation(solver.SPHIntegration

			sph.NBodyForce.withargs(),
			on_types=[Fluid], from_types=[Fluid],
			updates=['u','v','w'], id='nbody_force'

			)

In this example, the operation is of type **SPHIntegration** which uses the **NBodyForce** function to 
compute forces on all fluids from all fluids. Heruistically, an operation would be required for each
equation in the problem considered.

The final operation is therefore the position stepping operation which may have been declared as::

	s.add_operation(solver.SPHIntegration
	
			sph.PositionStepping.withargs(),
			on_types=[Fluid], updates=['x','y','z'],
			id='step'
			
			)

We achieve this by using the statement::

	s.to_step(types=[Fluid])

Once the operations are defined, we tell the solver to setup the integrator, setup time stepping information
and solve the system. Look at the examples for more information on using PySPH to solve your problems!



