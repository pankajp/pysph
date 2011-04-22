#include "cl_common.h"

// set the tmpx, tmpy, tmpz to 0.0
__kernel void set_tmp_to_zero(__global float* tmpx,
			      __global float* tmpy,
			      __global float* tmpz)
{
  unsigned int work_dim = get_work_dim();
  unsigned int gid = get_gid(work_dim);

  tmpx[gid] = 0.0f;
  tmpy[gid] = 0.0f;
  tmpz[gid] = 0.0f;

}

