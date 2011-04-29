
$GravityForce

#include "cl_common.h"
#include "cl_common.cl"
#include "kernels.h"

__kernel void GravityForce(%(kernel_args)s)
{
    %(workgroup_code)s

    tmpx[dest_id] = gx;
    tmpy[dest_id] = gy;
    tmpz[dest_id] = gz;
  
} // __kernel GravityForce
$GravityForce

$NBodyForce

#include "cl_common.h"
#include "cl_common.cl"
#include "kernels.h"

__kernel void NBodyForce(%(kernel_args)s)
{
    %(workgroup_code)s

    // The term `dest_id` will be suitably defined at this point.

    REAL4 pa = (REAL4)( d_x[dest_id],d_y[dest_id], d_z[dest_id], d_h[dest_id] );
    REAL4 rba;
    REAL invr, force_mag;

    %(neighbor_loop_code)s 
    {
        // SPH innermost loop code goes here.  The index `src_id` will
        // be available and looped over, this index.

        REAL4 pb = (REAL4)( s_x[src_id],s_y[src_id],s_z[src_id],s_h[src_id] );
        rba = pb - pa;
      
        invr = 1.0F/( length(rba) + eps );
        invr *= ( invr * invr );
	      
	force_mag = s_m[src_id] * invr;

	if ( self == TRUE )
	  {
	    if ( src_id != dest_id )
	      {
		tmpx[dest_id] += force_mag * rba.x;
		tmpy[dest_id] += force_mag * rba.y;
		tmpz[dest_id] += force_mag * rba.z;

	      } // if (i != dest_id)

	  } // if (self == true)

	else
	  {
	    tmpx[dest_id] += force_mag * rba.x;
	    tmpy[dest_id] += force_mag * rba.y;
	    tmpz[dest_id] += force_mag * rba.z;
	  }
	
    } // neighbor loop

} // __kernel NBodyForce
$NBodyForce     
