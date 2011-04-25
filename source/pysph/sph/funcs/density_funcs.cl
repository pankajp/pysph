#include "cl_common.h"
#include "cl_common.cl"
#include "kernels.h"

  
__kernel void SPHRho(int const kernel_type, int const dim, int const nbrs,
		     __global REAL* d_x, __global REAL* d_y, 
		     __global REAL* d_z, __global REAL* d_h,
		     __global int* tag,
		     __global REAL* s_x, __global REAL* s_y,
		     __global REAL* s_z, __global REAL* s_h,
		     __global REAL* s_m, __global REAL* s_rho,
		     __global REAL* tmpx)

{
  unsigned int work_dim = get_work_dim();
  unsigned int gid = get_gid(work_dim);
  
  REAL4 pa = (REAL4)( d_x[gid], d_y[gid], d_z[gid], d_h[gid] );
  REAL wmb = 0.0;
  REAL w;

  for (unsigned int i = 0; i < nbrs; ++i)
    {
      REAL4 pb = (REAL4)( s_x[i], s_y[i], s_z[i], s_h[i] );
      REAL mb = s_m[i];
      
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
