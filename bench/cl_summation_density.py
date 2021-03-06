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
    for device in devices:
        ctx = cl.Context([device])
        print("Device name:", device.name)
        print("Device type:", cl.device_type.to_string(device.type))
        print("Device memory: ", device.global_mem_size//1024//1024, 'MB')
        print("Device max clock speed:", device.max_clock_frequency, 'MHz')
        print("Device compute units:", device.max_compute_units)

        precision_types = ['single']
        
        device_extensions = device.get_info(cl.device_info.EXTENSIONS)
        if 'cl_khr_fp64' in device_extensions:
            precision_types.append('double')
            
        for prec in precision_types:

            print "\n========================================================"
            print "Summation Density Comparison using %s precision"%(prec)
        
            pa = base.get_particle_array(cl_precision=prec,
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
            
            cl_rho = pa.get('_tmpx').copy()
            
            # Do the same thing with Cython.
            t1 = time.time()
            calc.sph('_tmpx')
            cython_elapsed = time.time() - t1
            print "Execution time for PySPH Cython: %g s" %(cython_elapsed)

            # Compare the results

            cython_rho = pa.get('_tmpx')
            diff = sum(abs(cl_rho - cython_rho))
            
            if diff/np < 1e-6:
                print "CL == Cython: True"
                print "Speedup: %g "%(cython_elapsed/cl_elapsed)
            else:
                print "Results Don't Match!"

