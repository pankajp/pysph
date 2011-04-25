#include "cl_common.h"

// c[gid] = a[gid] + h*b[gid]
__kernel void step(__global REAL* a, __global REAL* b, __global REAL* c,
		   REAL const h)
{
  unsigned int work_dim = get_work_dim();
  unsigned int gid = get_gid(work_dim);

  c[gid] = a[gid] + h * b[gid];

}
