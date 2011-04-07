#include "cl_common.h"
#include "kernels.h"

__kernel void CL_SPHRho(__global float16* dst, __global float16* src,
			__global int* dst_tag,  __global float* result,
			__global int* kernel_type,  __global int* dim, 
			__global int* np)
{
  // x, y, z, u, v, w, h, m, rho, p, e, cs ?  ?  ?  ?
  // 0  1  2  3  4  5  6  7  8    9  10 11 12 13 14 15

  unsigned int work_dim = get_work_dim();

  unsigned int gid; // = get_global_id(0);
  
  Size local_size, num_groups, local_id, group_id;
  
  get_group_information(work_dim, &local_size, &num_groups, &local_id,
  			&group_id);

  unsigned int group_size = prod( &local_size );
  unsigned int ngroups = prod( &num_groups );

  gid = group_id.sizes[0] * group_size +
    group_id.sizes[1] * num_groups.sizes[0]*group_size +
    group_id.sizes[2] * num_groups.sizes[0]*num_groups.sizes[1]*group_size;

  gid += local_id.sizes[0] +
    local_id.sizes[1] * local_size.sizes[0] +
    local_id.sizes[2] * local_size.sizes[0]*local_size.sizes[1];

  float4 pa = (float4)( dst[gid].s0, dst[gid].s1, dst[gid].s2, dst[gid].s6 );
  float wmb = 0.0f;
  float w;

  int _dim = dim[0];
  int _np = np[0];

  for (unsigned int i = 0; i < _np; ++i)
    {
      float4 pb = (float4)( src[i].s0, src[i].s1, src[i].s2, src[i].s6 );
      float mb = src[i].s7;
      
      switch (kernel_type[0])
	{
	case 1:
	  w = cubic_spline_function(pa, pb, _dim);
	  break;
	  
	default:
	  w = cubic_spline_function(pa, pb, _dim);
	  break;
	      
	} // switch
	  
      wmb += w*mb;
	  	  
    } // for i
  
  result[gid] = wmb;

} // __kernel CL_SPHRho
