#include "cl_common.h"

// set the tmpx, tmpy, tmpz (sc, sd, se)  for the particles to 0
__kernel void set_tmp_to_zero(__global float16* dst)
{
  unsigned int work_dim = get_work_dim();

  Size local_size, num_groups, local_id, group_id;
  
  get_group_information(work_dim, &local_size, &num_groups, &local_id,
  			&group_id);

  unsigned int group_size = prod( &local_size );

  unsigned int gid = get_gid(&group_id, &num_groups, &local_id, &local_size,
			     group_size);

  dst[gid].sc = 0.0f;
  dst[gid].sd = 0.0f;
  dst[gid].se = 0.0f;
}
