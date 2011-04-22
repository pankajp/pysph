#define F f

#include "cl_common.h"
#include "cl_common.cl"
#include "kernels.h"
  
__kernel void SPHRho(int const kernel_type, int const dim, int const nbrs,
		     __global float* d_x, __global float* d_y, 
		     __global float* d_z, __global float* d_h,
		     __global int* tag,
		     __global float* s_x, __global float* s_y,
		     __global float* s_z, __global float* s_h,
		     __global float* s_m, __global float* s_rho,
		     __global float* tmpx)

{
  unsigned int work_dim = get_work_dim();
  unsigned int gid = get_gid(work_dim);
  
  float4 pa = (float4)( d_x[gid], d_y[gid], d_z[gid], d_h[gid] );
  float wmb = 0.0F;
  float w;

  for (unsigned int i = 0; i < nbrs; ++i)
    {
      float4 pb = (float4)( s_x[i], s_y[i], s_z[i], s_h[i] );
      float mb = s_m[i];
      
      switch (kernel_type)
	{
	case 1:
	  w = cubic_spline_function(pa, pb, dim);
	  break;
	  
	default:
	  w = cubic_spline_function(pa, pb, dim);
	  break;
	      
	} // switch
	  
      wmb += w*mb;
  	  	  
    } // for i

  tmpx[gid] += wmb;
  
} // __kernel SPHRho
