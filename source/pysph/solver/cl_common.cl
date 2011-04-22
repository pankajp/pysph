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

__kernel void test(__global float* a, __global float* b)
{
    unsigned int work_dim = get_work_dim();

  Size local_size, num_groups, local_id, group_id;
  
  get_group_information(work_dim, &local_size, &num_groups, &local_id,
  			&group_id);

  unsigned int group_size = prod( &local_size );

  unsigned int gid = get_gid(&group_id, &num_groups, &local_id, &local_size,
			     group_size);

  a[gid] = 10.0f;
  b[gid] = 20.0f;
}
