// OpenCL definitions of the kernels

#define PI asin(0)
#define INVPI 1.0f/PI

float norm(float4 vec)  
{  
  return sqrt( *(vec).s0 * *(vec).s0 + *(vec).s1 * *(vec).s1 + 
	       *(vec).s2 * *(vec).s2 );
}

float cubic_spline_function(float4* pa, float4* pb, unsigned int dim)
{
  float4 rab = *(pa) - *(pb);

  float h = 0.5f * ( *(pa).s3 + *(pb).s3 );
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

void cubic_spline_gradient(float4* pa, float4* pb, float4* grad, 
			   unsigned int dim)
{
  float4 rab = *(pa) - *(pb);

  float h = 0.5f * ( *(pa).s3 + *(pb).s3 );
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

  *(grad).s0 = val * rab.s0;
  *(grad).s1 = val * rab.s1;
  *(grad).s2 = val * rab.s2;

}
