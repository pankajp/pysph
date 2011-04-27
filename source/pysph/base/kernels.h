// C declarations for the SPH smoothing kernels
#ifndef CL_KERNEL_H
#define CL_KERNEL_H

#define PI 2.0F*acos(0.0f)

enum KernelType {
    CUBIC_SPLINE = 1,
    GAUSSIAN = 2,
    QUINTIC_SPLINE = 3,
    WENDLAND_QUINTIC_SPLINE = 4,
    HARMONIC = 5,
    M6_SPLINE = 6,
    W8 = 7,
    W10 = 8,
    REPULSIVE = 9,
    POLY6 = 10
};


REAL cubic_spline_fac(unsigned int dim, REAL h)
{
  REAL fac = 0.0F;

  if ( dim == 1 )
    fac = ( 2.0F/3.0F ) / (h);

  if (dim == 2 )
    fac = ( 10.0F ) / ( 7.0F*PI ) / ( h*h );

  if ( dim == 3 )
    fac = 1.0F /( PI*h*h*h );

  return fac;
}

REAL cubic_spline_function(REAL4 pa, REAL4 pb, unsigned int dim)
{
  // {s0, s1, s2, s3} == {x, y, z, h}

  REAL4 rab = pa - pb;

  REAL h = 0.5F * ( pa.s3 + pb.s3 );
  REAL q = length(rab)/h;

  REAL val = 0.0F;
  REAL fac = cubic_spline_fac(dim, h);
    
  if ( q >= 0.0F )
    val = 1.0F - 1.5F * (q*q) * (1.0F - 0.5F * q);

  if ( q >= 1.0F )
    val = 0.25F * (2.0F-q) * (2.0F-q) * (2.0F-q);

  if ( q >= 2.0F )
      val = 0.0F;
  
  return val * fac;
}

void cubic_spline_gradient(REAL4 pa, REAL4 pb, REAL4* grad, 
			   unsigned int dim)
{
  REAL4 rab = pa - pb;

  REAL h = 0.5F * ( pa.s3 + pb.s3 );
  REAL q = length(rab)/h;

  REAL val = 0.0F;
  REAL fac = cubic_spline_fac(dim, h);

  if ( q == 0.0F )
    val = 0.0F;

  if ( q > 0.0F )
    val = 3.0F*(0.75F*q - 1.0F)/(h*h);
  
  if ( q >= 1.0F )
    val = -0.75F * (2.0F-q) * (2.0F-q)/(h * length(rab));
      
  if ( q >= 2.0F )
    val = 0.0F;
  
  val *= fac;

  grad[0].s0 = val * rab.s0;
  grad[0].s1 = val * rab.s1;
  grad[0].s2 = val * rab.s2;

}

REAL kernel_function(REAL4 pa, REAL4 pb, unsigned int dim, int kernel_type)
{
    REAL w;

    switch (kernel_type) {
        case CUBIC_SPLINE:
            w = cubic_spline_function(pa, pb, dim);
            break;
        // FIXME: implement the rest!
        default:
            w = cubic_spline_function(pa, pb, dim);
            break;
    } // switch (kernel_type).

    return w;
}

void kernel_gradient(REAL4 pa, REAL4 pb, REAL4* grad, 
                     unsigned int dim, int kernel_type)
{
    switch (kernel_type) {
        case CUBIC_SPLINE:
            cubic_spline_gradient(pa, pb, grad, dim);
            break;
        // FIXME: implement the rest!
        default:
            cubic_spline_gradient(pa, pb, grad, dim);
            break;
    } // switch (kernel_type).
}

#endif // CL_KERNEL_H

