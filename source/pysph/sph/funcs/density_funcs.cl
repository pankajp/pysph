
// The kernel arguments are filled in automatically.
$SPHRho

#include "cl_common.h"
#include "cl_common.cl"
#include "kernels.h"

__kernel void SPHRho(%(kernel_args)s)
{
    %(workgroup_code)s

    // The term `dest_id` will be suitably defined at this point.

    REAL4 pa = (REAL4)( d_x[dest_id], d_y[dest_id], 
                        d_z[dest_id], d_h[dest_id] );
    REAL wmb = 0.0F ;
    REAL w;

    %(neighbor_loop_code)s 
    {
        // SPH innermost loop code goes here.  The index `src_id` will
        // be available and looped over, this index.

        REAL4 pb = (REAL4)(s_x[src_id], s_y[src_id], s_z[src_id], s_h[src_id]);
        REAL mb = s_m[src_id];
        w = kernel_function(pa, pb, dim, kernel_type); 
        wmb += w*mb;  	  
    }

    tmpx[dest_id] += wmb;
  
} // __kernel SPHRho

$SPHRho
