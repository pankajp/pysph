#define PI asin(0)
#define INVPI 1.0f/PI

typedef struct {
  float4 point;
} Point;

float norm (Point* p)
{  return sqrt(p->point.s0*p->point.s0 + 
	       p->point.s1*p->point.s1 + 
	       p->point.s2*p->point.s2);
}

typedef struct {
  int sizes[3];
} Size;

unsigned int prod(Size* size)
{
  return size->sizes[0] * size->sizes[1] * size->sizes[2];
}

float cubic_spline_function(Point* pa, Point* pb, unsigned int dim)
{
  Point rab;
  rab.point = pa->point - pb->point;

  float h = 0.5f * (pa->point.s3 + pb->point.s3);
  float q = norm(&rab)/h;

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

void cubic_spline_gradient(Point* pa, Point* pb, Point* grad, unsigned int dim)
{
  Point rab;
  rab.point = pa->point - pb->point;  

  float h = 0.5f * (pa->point.s3 + pb->point.s3);
  float q = norm(&rab)/h;

  float val = 0.0f;
  float fac = 0.0f;

  if ( dim == 1 )
    fac = ( 2.0f/3.0f ) / (h);

  if (dim == 2 )
    fac = ( 10.0f ) / ( 7.0f*PI ) / ( h*h );

  if ( dim == 3 )
    fac = INVPI / ( h*h*h );

  if ( q == 0.0f )
    val = 0.0f;

  if ( q > 0.0f )
    val = 3.0f*(0.75f*q - 1)/(h*h);
  
  if ( q >= 1.0f )
    val = -0.75f * (2.0f-q) * (2.0f-q)/(h * norm(&rab));

      
  if ( q >= 2.0f )
    val = 0.0f;
  
  val *= fac;

  grad->point.s0 = val * rab.point.s0;
  grad->point.s1 = val * rab.point.s1;
  grad->point.s2 = val * rab.point.s2;

}
  
__kernel void summation_density(__global const float4* dst,
				__global const float4* src,
				__global float* rho)
{
  
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

  Point pa, pb;
  float w = 0.0f;

  pa.point = dst[gid];
  for (unsigned int i = 0; i < 101; ++i)
    {
      pb.point = src[i];
      w += cubic_spline_function(&pa, &pb, 1);
    }

  rho[gid] = w;

}

// Testing kernels

__kernel void test_point_norm(__global float4* point_in, 
			      __global float* _norm)
{
  Point p;  
  p.point = point_in[0];  
  _norm[0] = norm(&p);

}

__kernel void test_cubic_spline_function(__global float4* pa,
					 __global float4* pb,
					 __global float* w)
{
  Point _pa, _pb;
  _pa.point = pa[0];
  _pb.point = pb[0];
  
  w[0] = cubic_spline_function(&_pa, &_pb, 1);
}

__kernel void test_cubic_spline_gradient(__global float4* pa,
					 __global float4* pb,
					 __global float4* grad)
{  
  Point _pa, _pb, _grad;
  _pa.point = pa[0];
  _pb.point = pb[0];
  _grad.point = grad[0];

  cubic_spline_gradient(&_pa, &_pb, &_grad, 1);

  grad[0] = _grad.point;
}
