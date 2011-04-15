#include "cl_common.h"
#include "cl_common.cl"
#include "kernels.h"

__kernel void SPHRho(__global float16* dst, __global float16* src,
		     __global int* dst_tag, __global int* np,
		     __global int* kernel_type,  __global int* dim )

{
  // x, y, z, u, v, w, h, m, rho, p, e, cs tmpx  tmpy  tmpz  x
  // 0  1  2  3  4  5  6  7  8    9  a  b   c     d     e    f

  unsigned int work_dim = get_work_dim();

  Size local_size, num_groups, local_id, group_id;
  
  get_group_information(work_dim, &local_size, &num_groups, &local_id,
  			&group_id);

  unsigned int group_size = prod( &local_size );

  unsigned int gid = get_gid(&group_id, &num_groups, &local_id, &local_size,
			     group_size);

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

  dst[gid].sc += wmb;
  
} // __kernel SPHRho_CL
