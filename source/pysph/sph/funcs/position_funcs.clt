
$PositionStepping

#include "cl_common.h"
#include "cl_common.cl"
#include "kernels.h"

__kernel void PositionStepping(__global float* d_u, __global float* d_v,
			       __global float* d_w, __global int* tag,
			       __global float* tmpx, __global float* tmpy,
			       __global float* tmpz )
{
  unsigned int work_dim = get_work_dim();
  unsigned int gid = get_gid(work_dim);

  tmpx[gid] = d_u[0];
  tmpy[gid] = d_v[0];
  tmpz[gid] = d_w[0];

}
$PositionStepping
