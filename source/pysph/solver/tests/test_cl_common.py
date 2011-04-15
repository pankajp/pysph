""" Tests for the OpenCL functions in cl_common """
import pyopencl as cl
from pyopencl.array import vec

import numpy
import pysph.solver.api as solver

from os import path

np = 16*16*16

x = numpy.ones(np, numpy.float32)
y = numpy.ones(np, numpy.float32)
z = numpy.ones(np, numpy.float32)

platform = cl.get_platforms()[0]
devices = platform.get_devices()
device = devices[0]

ctx = cl.Context(devices)
q = cl.CommandQueue(ctx, device)

mf = cl.mem_flags

xbuf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=x)
ybuf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=y)
zbuf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=z)

args = (ybuf, zbuf)

pysph_root = solver.get_pysph_root()
src = open(path.join(pysph_root, 'solver/cl_common.cl')).read()

prog = cl.Program(ctx, src).build(options=solver.get_cl_include())

# launch the OpenCL kernel
prog.set_tmp_to_zero(q, (16, 16, 16), (1,1,1), xbuf, *args)

# read the buffer contents back to the arrays
cl.enqueue_read_buffer(q, xbuf, x).wait()
cl.enqueue_read_buffer(q, ybuf, y).wait()
cl.enqueue_read_buffer(q, zbuf, z).wait()

for i in range(np):
    assert x[i] == 0.0
    assert y[i] == 0.0
    assert z[i] == 0.0

print "OK"
