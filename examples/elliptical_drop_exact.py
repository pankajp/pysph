def exact_solution(tf=0.00076, dt=1e-4):
    """ Exact solution for the the elliptical drop equations """
    import numpy
    
    A0 = 100
    a0 = 1.0

    t = 0.0

    theta = numpy.linspace(0,2*numpy.pi, 101)

    Anew = A0
    anew = a0

    while t <= tf:
        t += dt

        Aold = Anew
        aold = anew
        
        Anew = Aold +  dt*(Aold*Aold*(aold**4 - 1))/(aold**4 + 1)
        anew = aold +  dt*(-aold * Aold)

    dadt = Anew**2 * (anew**4 - 1)/(anew**4 + 1)
    po = 0.5*-anew**2 * (dadt - Anew**2)
        
    return anew*numpy.cos(theta), 1/anew*numpy.sin(theta), po


#############################################################################
