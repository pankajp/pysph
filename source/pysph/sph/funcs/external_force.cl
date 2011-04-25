#include "cl_common.h"
#include "cl_common.cl"
#include "kernels.h"
  
__kernel void GravityForce(REAL const gx, REAL const gy, REAL const gz,
			   __global REAL* tmpx, __global REAL* tmpy,
			   __global REAL* tmpz)
{
  unsigned int work_dim = get_work_dim();
  unsigned int gid = get_gid(work_dim);

  tmpx[gid] = gx;
  tmpy[gid] = gy;
  tmpz[gid] = gz;
  
} // __kernel GravityForce


__kernel void NBodyForce(int const nbrs, int const self,
			 REAL const eps,
			 __global REAL* d_x, __global REAL* d_y, 
			 __global REAL* d_z, __global REAL* d_h,
			 __global int* tag,
			 __global REAL* s_x, __global REAL* s_y,
			 __global REAL* s_z, __global REAL* s_h,
			 __global REAL* s_m, __global REAL* s_rho,
			 __global REAL* tmpx, __global REAL* tmpy,
			 __global REAL* tmpz)

{
  unsigned int work_dim = get_work_dim();
  unsigned int gid = get_gid(work_dim);

  REAL4 pa = (REAL4)( d_x[gid], d_y[gid], d_z[gid], d_h[gid] );
  REAL4 rba;

  REAL invr, force_mag;

  for (unsigned int i = 0; i < nbrs; ++i)
    {
      
      REAL4 pb = (REAL4)( s_x[i], s_y[i], s_z[i], s_h[i] );
      rba = pb - pa;
      
      invr = 1.0/( length(rba) + eps );
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
     
