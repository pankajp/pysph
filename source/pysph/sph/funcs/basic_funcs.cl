#define PI asin(0)
#define INVPI 1.0f/PI

typedef struct {
  float4 point;
} Point;

float norm (float4 p)
{  return sqrt( p.s0*p.s0 +  p.s1*p.s1 + p.s2*p.s2 );
}

typedef struct {
  int sizes[3];
} Size;

unsigned int prod(Size* size)
{
  return size->sizes[0] * size->sizes[1] * size->sizes[2];
}

float cubic_spline_function(float4 pa, float4 pb, unsigned int dim)
{
  float4 rab = pa - pb;

  float h = 0.5f * (pa.s3 + pb.s3);
  float q = norm(rab)/h;

  float val = 0.0f;
  float fac = 0.0f;

  if ( dim == 1 )
    fac = ( 2.0f/3.0f ) / (h);

  if (dim == 2 )
    fac = ( 10.0f ) / ( 7*PI ) / ( h*h );

  if ( dim == 3 )
    fac = INVPI / ( h*h*h );
    
  if ( q >= 0.0f )
    val = 1.0f - 1.5f * (q*q) * (1.0f - 0.5f * q);

  if ( q >= 1.0f )
    val = 0.25f * (2.0f-q) * (2.0f-q) * (2.0f-q);

  if ( q >= 2.0f )
      val = 0.0f;
  
  return val * fac;

}

__kernel void SPH(__global float16* dst, __global float16* src,
		  __global int* dst_tag,  __global float* result,
		  __global int* kernel_type,  __global int* prop_num,
		  __global int* np, __global int* dim)
{
  // x, y, z, u, v, w, h, m, rho, p, e, cs ?  ?  ?  ?
  // 0  1  2  3  4  5  6  7  8    9  10 11 12 13 14 15

  unsigned int work_dim = get_work_dim();

  unsigned int gid;
  
  Size local_size, num_groups, local_id, group_id;
  
  for (unsigned int i = 0; i < work_dim; ++i)
    {
      local_size.sizes[i] = get_local_size(i);
      num_groups.sizes[i] = get_num_groups(i);
      local_id.sizes[i] = get_local_id(i);
      group_id.sizes[i] = get_group_id(i);

    } // for i

  unsigned int group_size = prod( &local_size );
  unsigned int ngroups = prod( &num_groups );

  gid = group_id.sizes[0] * group_size +
    group_id.sizes[1] * num_groups.sizes[0]*group_size +
    group_id.sizes[2] * num_groups.sizes[0]*num_groups.sizes[1]*group_size;

  gid += local_id.sizes[0] + 
    local_id.sizes[1] * local_size.sizes[0] +
    local_id.sizes[2] * local_size.sizes[0]*local_size.sizes[1];

  float4 pa = (float4)( dst[gid].s0, dst[gid].s1, dst[gid].s2, dst[gid].s6 );
  float w;

  if (dst_tag[gid] == 1 )
    {

      for (unsigned int i = 0; i < np[0]; i++)
	{
	  float4 pb = (float4)( src[i].s0, src[i].s1, src[i].s2, src[i].s6 );
	  float rhob = src[i].s8;
	  float mb = src[i].s7;
	  float fb = src[i].s8;
      
	  switch (kernel_type[0])
	    {
	    case 1:
	      w = cubic_spline_function(pa, pb, dim[0]);
	      break;
	
	    default:
	      w = cubic_spline_function(pa, pb, dim[0]);
	      break;
	      
	    } // switch
	  
	  result[gid] += fb*mb*w/rhob;
	  
	  
	} // for i

    } // if dst_tag

} // __kernel
