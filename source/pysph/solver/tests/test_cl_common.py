""" Tests for the OpenCL functions in cl_common """
import pyopencl as cl
from pyopencl.array import vec

import numpy
import pysph.solver.api as solver

from os import path
import time

np = 256*256*16

x = numpy.zeros((np,), vec.float16)

for i in range(np):
    x[i][12] = 10
    x[i][13] = 20
    x[i][14] = 30

platform = cl.get_platforms()[0]
devices = platform.get_devices()
device = devices[0]

ctx = cl.Context(devices)
q = cl.CommandQueue(ctx, device)

mf = cl.mem_flags

xbuf = cl.Buffer(ctx, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=x)

pysph_root = solver.get_pysph_root()
src = open(path.join(pysph_root, 'solver/cl_common.cl')).read()

prog = cl.Program(ctx, src).build(options=solver.get_cl_include())

t1 = time.time()
prog.set_to_zero(q, (256, 256, 16), (1,1,1), xbuf)
print "Time taken: ", time.time() - t1

out = numpy.empty(x.shape, x.dtype)
cl.enqueue_read_buffer(q, xbuf, out).wait()

for i in range(np):
    assert out[i][12] == 0.0
    assert out[i][13] == 0.0
    assert out[i][14] == 0.0
