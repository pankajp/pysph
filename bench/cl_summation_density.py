import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph

import numpy
import time
import pyopencl as cl

NSquareNeighborLocator = base.NeighborLocatorType.NSquareNeighborLocator

np = 5001

x = numpy.linspace(0,1,np)
m = numpy.ones_like(x) * (x[1] - x[0])
h = 2*m
rho = numpy.ones_like(x)

# get the OpenCL context and device. Default to the first device

platforms = cl.get_platforms()
for platform in platforms:
    print("===============================================================")
    print("Platform name:", platform.name)
    print("Platform profile:", platform.profile)
    print("Platform vendor:", platform.vendor)
    print("Platform version:", platform.version)
    print("---------------------------------------------------------------")
    devices = platform.get_devices()
    ctx = cl.Context(devices)
    for device in devices:
        print("Device name:", device.name)
        print("Device type:", cl.device_type.to_string(device.type))
        print("Device memory: ", device.global_mem_size//1024//1024, 'MB')
        print("Device max clock speed:", device.max_clock_frequency, 'MHz')
        print("Device compute units:", device.max_compute_units)

        # create the partice array with doubles as the default

        pa = base.get_particle_array(floating_point_default="double",
                                     name="test", x=x,h=h,m=m,rho=rho)

        particles = base.Particles(arrays=[pa,] ,
                                   locator_type=NSquareNeighborLocator)

        kernel = base.CubicSplineKernel(dim=1)

        # create the function
        func = sph.SPHRho.get_func(pa,pa)

        # create the CLCalc object
        cl_calc = sph.CLCalc(particles=particles, sources=[pa,], dest=pa,
                             kernel=kernel, funcs=[func,], updates=['rho'] )

        # create a normal calc object
        calc = sph.SPHCalc(particles=particles, sources=[pa,], dest=pa,
                           kernel=kernel, funcs=[func,], updates=['rho'] )


        # setup OpenCL for PySPH
        cl_calc.setup_cl(ctx)

        # evaluate pysph on the OpenCL device!
        t1 = time.time()
        cl_calc.sph()
        cl_elapsed = time.time() - t1
        print "Execution time for PyOpenCL: %g s" %(cl_elapsed)

        # Read the buffer contents
        t1 = time.time()
        pa.read_from_buffer()
        read_elapsed = time.time() - t1

        print "Read from buffer time: %g s "%(read_elapsed)

        cl_rho = pa.get('tmpx').copy()

        # Do the same thing with Cython.
        t1 = time.time()
        calc.sph('tmpx')
        cython_elapsed = time.time() - t1
        print "Execution time for PySPH Cython: %g s" %(cython_elapsed)

        # Compare the results

        diff = 0.0
        cython_rho = pa.get('tmpx')
        for i in range(np):
            diff += abs( cl_rho[i] - cython_rho[i] )
            
        if diff/np < 1e-6:
            print "CL == Cython: True"
            print "Speedup: %g "%(cython_elapsed/cl_elapsed)
        else:
            print "Results Don't Match!"

