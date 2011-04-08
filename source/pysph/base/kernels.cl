#include "kernels.h"

__kernel void cl_cubic_spline_fac(__global float* h, 
				  __global unsigned int* dim,
				  __global float* fac)
{
  float _h = h[0];
  unsigned int _dim = dim[0];

  float _fac = cubic_spline_fac(_dim, _h);
  fac[0] = _fac;

}
__kernel void cl_cubic_spline_function(__global float4* pa, __global float4* pb,
				       __global float* w, 
				       __global unsigned int* dim)
{
  float4 _pa = pa[0];
  float4 _pb = pb[0];
  unsigned int _dim = dim[0];

  float _w = cubic_spline_function(_pa, _pb, _dim);
  w[0] = _w;
}


__kernel void cl_cubic_spline_gradient(__global float4* pa, __global float4* pb,
				       __global float4* grad, 
				       __global unsigned int* dim)
{
  float4 _pa = pa[0];
  float4 _pb = pb[0];
  float4 _grad = grad[0];
  unsigned int _dim = dim[0];

  cubic_spline_gradient(_pa, _pb, &_grad, _dim);

  grad[0] = _grad;
}
