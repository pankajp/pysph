import tempfile
import os

import pysph.solver.cl_utils as clu

def test_cl_read():
    """Test if the pysph.solcer.cl_utils.cl_read works."""

    # Create a test file.
    fd, name = tempfile.mkstemp(suffix='.cl')
    code = """
    REAL foo = 1.0;
    """
    f = open(name, 'w')
    f.write(code)
    f.close()
    os.close(fd)

    # Test single precision
    src = clu.cl_read(name, precision='single')
    expect = """#define F f
#define REAL float
#define REAL2 float2
#define REAL3 float3
#define REAL4 float4
#define REAL8 float8
"""
    s_lines = src.split()
    for idx, line in enumerate(expect.split()):
        assert line == s_lines[idx]


    # Test double precision
    src = clu.cl_read(name, precision='double')
    expect = """#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define F
#define REAL double
#define REAL2 double2
#define REAL3 double3
#define REAL4 double4
#define REAL8 double8
"""
    s_lines = src.split()
    for idx, line in enumerate(expect.split()):
        assert line == s_lines[idx]

    # cleanup.
    os.remove(name)


if __name__ == '__main__':
    test_cl_read()

