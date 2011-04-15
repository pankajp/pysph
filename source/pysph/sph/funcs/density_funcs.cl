#include "cl_common.h"
#include "cl_common.cl"
#include "kernels.h"
  
__kernel void SPHRho(__global int* kernel_type, __global int* dim,
		     __global int* nbrs,
		     __global float* d_x, __global float* d_y, 
		     __global float* d_z, __global float* d_h,
		     __global int* tag,
		     __global float* s_x, __global float* s_y,
		     __global float* s_z, __global float* s_h,
		     __global float* s_m, __global float* s_rho,
		     __global float* tmpx)

{
  unsigned int work_dim = get_work_dim();

  Size local_size, num_groups, local_id, group_id;
  
  get_group_information(work_dim, &local_size, &num_groups, &local_id,
  			&group_id);

  unsigned int group_size = prod( &local_size );

  unsigned int gid = get_gid(&group_id, &num_groups, &local_id, &local_size,
			     group_size);
  
  float4 pa = (float4)( d_x[gid], d_y[gid], d_z[gid], d_h[gid] );
  float wmb = 0.0f;
  float w;

  int _dim = dim[0];
  int _np = nbrs[0];

  for (unsigned int i = 0; i < _np; ++i)
    {
      float4 pb = (float4)( s_x[i], s_y[i], s_z[i], s_h[i] );
      float mb = s_m[i];
      
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

  tmpx[gid] += wmb;
  
} // __kernel SPHRho
