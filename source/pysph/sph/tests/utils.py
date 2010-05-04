import numpy 

def generate_square(pnt1, pnt2, dx, dy):
    """ Generate a square from the given displacements. """
    
    x0 = pnt1[0]; x1 = pnt2[0]
    y0 = pnt1[1]; y1 = pnt2[1]

    xa = numpy.arange(x0, x1, dx)
    ya = numpy.arange(y0, y1, dy)

    nx = len(xa); ny = len(ya)
    np = nx * ny

    x = numpy.zeros(np, float)
    y = numpy.zeros(np, float)

    for i in range(ny):
        x[i*nx:(i+1)*nx] = xa
        y[i*nx:(i+1)*nx] = ya[i]
        
    return x, y

def _generate_circle(cen, rad, dr, dt, theta = 2*numpy.pi):
    """ Generate a circle in the plane given the displacements."""

    xc = cen[0]; yc = cen[1]
    t = numpy.arange(0, theta, dt)
    nt = len(t)
    r = numpy.linspace(xc+dr, xc + rad, nt)
    
    nr = len(r)
    np = nr * nt

    x = numpy.zeros(np, float)
    y = numpy.zeros(np, float)

    for i, j in enumerate(t):
        _x = r * numpy.cos(j)
        _y = r * numpy.sin(j)
        x[i*nr:(i+1)*nr][:] = _x
        y[i*nr:(i+1)*nr][:] = _y

    return x, y

"""

x1, y1 = generate_square((-.25, -.25), (.25, .25), .05,.05)
x2, y2 = generate_square((.25, .25), (.5, .5), .05,.05)
x3, y3 = generate_square((.25, -.25), (.5, .25), .05,.05)
x4, y4 = generate_square((.25, -.5), (.5, -.25), .05,.05)
x5, y5 = generate_square((-.25, -.5), (.25, -.25), .05,.05)
x6, y6 = generate_square((-.5, -.5), (-.25, -.25), .05,.05)
x7, y7 = generate_square((-.5, -.25), (-.25, .25), .05,.05)
x8, y8 = generate_square((-.5, .25), (-.25, .5), .05,.05)
x9, y9 = generate_square((-.25, .25), (.25, .5), .05,.05)


plot(x1, y1, 'o', x2, y2, 'o', x3, y3, 'o', x4, y4,'o', x5, y5, 'o', x6, y6,'o')
plot(x7, y7, 'o', x8, y8, 'o', x9,y9, 'o')
show()

"""
