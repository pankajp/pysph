// C declarations for the SPH smoothing kernels

#define PI 2.0f*acos(0.0f)

REAL cubic_spline_fac(unsigned int dim, REAL h)
{
  REAL fac = 0.0;

  if ( dim == 1 )
    fac = ( 2.0/3.0 ) / (h);

  if (dim == 2 )
    fac = ( 10.0 ) / ( 7.0*PI ) / ( h*h );

  if ( dim == 3 )
    fac = 1.0 /( PI*h*h*h );

  return fac;
}

REAL cubic_spline_function(REAL4 pa, REAL4 pb, unsigned int dim)
{
  // {s0, s1, s2, s3} == {x, y, z, h}

  REAL4 rab = pa - pb;

  REAL h = 0.5 * ( pa.s3 + pb.s3 );
  REAL q = length(rab)/h;

  REAL val = 0.0;
  REAL fac = cubic_spline_fac(dim, h);
    
  if ( q >= 0.0 )
    val = 1.0 - 1.5 * (q*q) * (1.0 - 0.5 * q);

  if ( q >= 1.0 )
    val = 0.25 * (2.0-q) * (2.0-q) * (2.0-q);

  if ( q >= 2.0 )
      val = 0.0;
  
  return val * fac;
}

void cubic_spline_gradient(REAL4 pa, REAL4 pb, REAL4* grad, 
			   unsigned int dim)
{
  REAL4 rab = pa - pb;

  REAL h = 0.5 * ( pa.s3 + pb.s3 );
  REAL q = length(rab)/h;

  REAL val = 0.0;
  REAL fac = cubic_spline_fac(dim, h);

  if ( q == 0.0 )
    val = 0.0;

  if ( q > 0.0 )
    val = 3.0*(0.75*q - 1.0)/(h*h);
  
  if ( q >= 1.0 )
    val = -0.75 * (2.0-q) * (2.0-q)/(h * length(rab));
      
  if ( q >= 2.0 )
    val = 0.0;
  
  val *= fac;

  grad[0].s0 = val * rab.s0;
  grad[0].s1 = val * rab.s1;
  grad[0].s2 = val * rab.s2;

}
