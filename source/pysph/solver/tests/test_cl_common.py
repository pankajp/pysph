""" Tests for the OpenCL functions in cl_common """
import pyopencl as cl
from pyopencl.array import vec

import numpy
from pysph.solver.api import get_cl_include

np = 2
x = numpy.zeros((np,), vec.float16)

platform = cl.get_platforms()[0]
devices = platform.get_devices()
device = devices[0]

ctx = cl.Context(devices)
q = cl.CommandQueue(ctx, device)

mf = cl.mem_flags

xbuf = cl.Buffer(ctx, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=x)
src = open("/home/kunalp/pysph/source/pysph/solver/cl_common.cl").read()

prog = cl.Program(ctx, src).build(options=get_cl_include())

prog.set_to_zero(q, (np, 1, 1), (1,1,1), xbuf)

out = numpy.empty(x.shape, x.dtype)
cl.enqueue_read_buffer(q, xbuf, out).wait()

print out
