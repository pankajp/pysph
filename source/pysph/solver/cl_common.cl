#include "cl_common.h"

// set the tmpx, tmpy, tmpz to 0.0
__kernel void set_tmp_to_zero(__global REAL* tmpx,
			      __global REAL* tmpy,
			      __global REAL* tmpz)
{
  unsigned int work_dim = get_work_dim();
  unsigned int gid = get_gid(work_dim);

  tmpx[gid] = 0.0F;
  tmpy[gid] = 0.0F;
  tmpz[gid] = 0.0F;

}

