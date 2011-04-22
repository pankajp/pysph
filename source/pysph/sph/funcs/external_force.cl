#include "cl_common.h"
#include "cl_common.cl"
#include "kernels.h"
  
__kernel void GravityForce(float const gx, float const gy, float const gz,
			   __global float* tmpx, __global float* tmpy,
			   __global float* tmpz)
{
  unsigned int work_dim = get_work_dim();
  unsigned int gid = get_gid(work_dim);

  tmpx[gid] = gx;
  tmpy[gid] = gy;
  tmpz[gid] = gz;
  
} // __kernel GravityForce


__kernel void NBodyForce(int const nbrs, int const self,
			 float const eps,
			 __global float* d_x, __global float* d_y, 
			 __global float* d_z, __global float* d_h,
			 __global int* tag,
			 __global float* s_x, __global float* s_y,
			 __global float* s_z, __global float* s_h,
			 __global float* s_m, __global float* s_rho,
			 __global float* tmpx, __global float* tmpy,
			 __global float* tmpz)

{
  unsigned int work_dim = get_work_dim();
  unsigned int gid = get_gid(work_dim);

  float4 pa = (float4)( d_x[gid], d_y[gid], d_z[gid], d_h[gid] );
  float4 rba;

  float invr, force_mag;

  for (unsigned int i = 0; i < nbrs; ++i)
    {
      
      float4 pb = (float4)( s_x[i], s_y[i], s_z[i], s_h[i] );
      rba = pb - pa;
      
      invr = 1.0f/( length(rba) + eps );
      invr *= ( invr * invr );
	      
      force_mag = s_m[i] * invr;

      if ( self == TRUE )
	{
	  if ( i != gid )
	    {
	      tmpx[gid] += force_mag * rba.x;
	      tmpy[gid] += force_mag * rba.y;
	      tmpz[gid] += force_mag * rba.z;

	    } // if (i != gid)

	} // if (self == true)

      else
	{
	  tmpx[gid] += force_mag * rba.x;
	  tmpy[gid] += force_mag * rba.y;
	  tmpz[gid] += force_mag * rba.z;
	}
      
    } // for i

} // __kernel NBodyForce
     
