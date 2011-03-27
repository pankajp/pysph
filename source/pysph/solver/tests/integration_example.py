from numpy import arccos, sin, cos, array, sqrt, pi

r = 2.0/pi
dt = 1e-3

def force(x,y):

    theta = arccos(x/sqrt((x**2+y**2)))
    return array([-sin(theta), cos(theta)])

def rk2(nsteps=1000, x0=r, y0=0):
    t = 0
    xinitial = x0
    yinitial = y0
    while t < nsteps:
        _x = xinitial
        _y = yinitial

        k1x, k1y = force(xinitial, yinitial)

        xinitial = _x + 0.5*dt*k1x; yinitial = _y + 0.5*dt*k1y
        k2x, k2y = force(xinitial, yinitial)

        xnew = _x + (0.5*dt)*(k1x + k2x)
        ynew = _y + (0.5*dt)*(k1y + k2y)

        xinitial = xnew
        yinitial = ynew
        t += 1
        pass
    
    return xnew, ynew

def rk4(steps=1000, x0=r, y0=0):
    t = 0
    xinitial = x0
    yinitial = y0
    while t < steps:
        _x = xinitial
        _y = yinitial
        
        k1x, k1y = force(xinitial, yinitial)

        xinitial = _x + 0.5*dt*k1x; yinitial = _y + 0.5*dt*k1y
        k2x, k2y = force(xinitial, yinitial)

        xinitial =_x +  0.5*dt*k2x; yinitial = _y + 0.5*dt*k2y
        k3x, k3y = force(xinitial, yinitial)

        xinitial = _x + dt*k3x; yinitial = _y + dt*k3y
        k4x, k4y = force(xinitial, yinitial)

        xnew = _x + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
        ynew = _y + (dt/6.0)*(k1y + 2*k2y + 2*k3y + k4y)

        xinitial = xnew
        yinitial = ynew

        t += 1
        pass
    return xnew, ynew


def euler(nsteps=1000, x0=r, y0=0):
    t = 0
    xinitial = x0
    yinitial = y0
    while t < nsteps:
        k1x, k1y =  dt*force(xinitial, yinitial)
        xnew = xinitial + k1x
        ynew = yinitial + k1y
        xinitial = xnew
        yinitial = ynew
        t += 1
        pass
    return xnew, ynew
