// C declarations for the SPH smoothing kernels

#define PI 2.0f*acos(0.0f)

float cubic_spline_fac(unsigned int dim, float h)
{
  float fac = 0.0f;

  if ( dim == 1 )
    fac = ( 2.0f/3.0f ) / (h);

  if (dim == 2 )
    fac = ( 10.0f ) / ( 7.0f*PI ) / ( h*h );

  if ( dim == 3 )
    fac = 1.0f /( PI*h*h*h );

  return fac;
}

float cubic_spline_function(float4 pa, float4 pb, unsigned int dim)
{
  // {s0, s1, s2, s3} == {x, y, z, h}

  float4 rab = pa - pb;

  float h = 0.5f * ( pa.s3 + pb.s3 );
  float q = length(rab)/h;

  float val = 0.0f;
  float fac = cubic_spline_fac(dim, h);
    
  if ( q >= 0.0f )
    val = 1.0f - 1.5f * (q*q) * (1.0f - 0.5f * q);

  if ( q >= 1.0f )
    val = 0.25f * (2.0f-q) * (2.0f-q) * (2.0f-q);

  if ( q >= 2.0f )
      val = 0.0f;
  
  return val * fac;
}

void cubic_spline_gradient(float4 pa, float4 pb, float4* grad, 
			   unsigned int dim)
{
  float4 rab = pa - pb;

  float h = 0.5f * ( pa.s3 + pb.s3 );
  float q = length(rab)/h;

  float val = 0.0f;
  float fac = cubic_spline_fac(dim, h);

  if ( q == 0.0f )
    val = 0.0f;

  if ( q > 0.0f )
    val = 3.0f*(0.75f*q - 1.0f)/(h*h);
  
  if ( q >= 1.0f )
    val = -0.75f * (2.0f-q) * (2.0f-q)/(h * length(rab));
      
  if ( q >= 2.0f )
    val = 0.0f;
  
  val *= fac;

  grad[0].s0 = val * rab.s0;
  grad[0].s1 = val * rab.s1;
  grad[0].s2 = val * rab.s2;

}
