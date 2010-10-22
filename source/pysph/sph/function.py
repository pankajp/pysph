""" Classes to get the various functions """
import funcs.basic_funcs as basic
import funcs.boundary_funcs as boundary
import funcs.density_funcs as density
import funcs.energy_funcs as energy
import funcs.eos_funcs as eos
import funcs.external_force as external
import funcs.position_funcs as position
import funcs.pressure_funcs as pressure
import funcs.viscosity_funcs as viscosity
import funcs.xsph_funcs as xsph

#base class
class Function(object):
    def __init__(self):
        pass

    def get_func(self, source, dst):
        raise NotImplementedError

#basic functions
class SPHInterpolation(Function):
    
    def __init__(self, prop_name=""):
        self.prop_name = prop_name
    
    def get_func(self, source, dest):
        return basic.SPH(source=source, dest=dest, prop_name=self.prop_name)
        
class SimpleDerivative(Function):
    
    def __init__(self, prop_name=""):
        self.prop_name = prop_name

    def get_func(self, source, dest):
        return basic.SPHSimpleDerivative(source=source, dest=dest, 
                                   prop_name=self.prop_name)
class Gradient(Function):
    
    def __init__(self, prop_name=""):
        self.prop_name = prop_name

    def get_func(self, source, dest):
        return basic.SPHGrad(source=source, dest=dest, prop_name=self.prop_name)

class Laplacian(Function):
    
    def __init__(self, prop_name=""):
        self.prop_name = prop_name

    def get_func(self, source, dest):
        return basic.SPHLaplacian(source, dest=dest, prop_name=self.prop_name)

class NeighborCount(Function):
    
    def get_func(self, source, dest):
        return basic.CountNeighbors(source=source, dest=dest)

#boundary functions

class MonaghanBoundaryForce(Function):
    
    def __init__(self, cs = -1, delp = 0.0):
        self.cs = cs
        self.delp = delp
        
    def get_func(self, source, dest):
        return boundary.MonaghanBoundaryForce(source=source, dest=dest,
                                              cs = self.cs, delp = self.delp)


class BeckerBoundaryForce(Function):
    
    def __init__(self, sound_speed):
        self.sound_speed = sound_speed

    def get_func(source, dest):
        return boundary.BeckerBoundaryForce(source=source, dest=dest,
                                            sound_speed=self.sound_speed)


class LennardJonesForce(Function):
    
    def __init__(self, D, ro, p1, p2):
        self.D = D
        self.ro = ro
        self.p1 = p1
        self.p2 = p2

    def get_func(self, source, dest):
        return boundary.LennardJonesForce(source=source, dest=dest,
                                          D=self.D, ro=self.ro, 
                                          p1=self.p1, p2=self.p2)

class SPHRho(Function):
    def get_func(self, source, dest):
        return density.SPHRho(source=source, dest=dest)

class SPHDensityRate(Function):
    def get_func(self, source, dest):
        return density.SPHDensityRate(source=source, dest=dest)


class EnergyEquationNoVisc(Function):
    
    def get_func(self, source, dest):
        return energy.EnergyEquation(source=source, dest=dest)

class EnergyEquationAVisc(Function):
    
    def __init__(self, beta=1.0, alpha=1.0, cs=-1, gamma=1.4, eta=0.1):
        self.beta = beta
        self.alpha = alpha
        self.cs = cs
        self.gamma = gamma
        self.eta = eta

    def get_func(self, source, dest):
        return energy.EnergyEquationAVisc(source=source, dest=dest, 
                                          beta=self.beta,
                                          gamma=self.gamma, alpha=self.alpha, 
                                          cs=self.cs, eta=self.eta)


class EnergyEquation(Function):
    
    def __init__(self, beta=1.0, alpha=1.0, gamma=1.4, eta=0.1):
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta        
        
    def get_func(self, source, dest):
        return energy.EnergyEquation(source=source, dest=dest, beta=self.beta,
                                     gamma=self.gamma, alpha=self.alpha, 
                                     eta=self.eta)


#state functions

class IdealGasEquation(Function):
    def __init__(self, gamma=1.4):
        self.gamma = gamma

    def get_func(self, source, dest):
        return eos.IdealGasEquation(source=source, dest=dest, gamma=self.gamma)

class TaitEquation(Function):
    def __init__(self, ko, ro, gamma=7):
        self.ko = ko
        self.ro = ro
        self.gamma = gamma

    def get_func(self, source, dest):
        return eos.TaitEquation(source=source, dest=dest, ko=self.ko, 
                                ro=self.ro, gamma=self.gamma)


class GravityForce(Function):
    def __init__(self, gx=0.0, gy=-9.81, gz=0.0):
        self.gx=gx
        self.gy=gy
        self.gz=gz

    def get_func(self, source, dest):
        return external.GravityForce(source=source, dest=dest, gx=self.gx,
                                     gy=self.gy, gz=self.gz)

class VectorForce(Function):
    def __init__(self, force):
        self.force = force

    def get_func(self, source, dest):
        return external.VectorForce(source=source, dest=dest, force=force)

class MoveCircleX(Function):
    def get_func(self, source, dest):
        return external.MoveCircleX(source=source, dest=dest)

class MoveCircleY(Function):
    def get_func(self, source, dest):
        return external.MoveCircleY(source=source, dest=dest)


#position funcs

class PositionStepping(Function):
    def get_func(self, source, dest):
        return position.PositionStepping(source=source, dest=dest)


class SPHPressureGradient(Function):
    def get_func(self, source, dest):
        return pressure.SPHPressureGradient(source=source, dest=dest)    

class MomentumEquation(Function):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.4, eta=0.1, 
                 sound_speed=-1):
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.eta=eta
        self.sound_speed=sound_speed

    def get_func(self, source, dest):
        return pressure.MomentumEquation(source=source, dest=dest,
                                         alpha=self.alpha,
                                         beta=self.beta, gamma=self.gamma, 
                                         eta=self.eta, 
                                         sound_speed=self.sound_speed)

class MonaghanArtificialVsicosity(Function):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.4, eta=0.1, c=-1):
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.eta=eta
        self.c = c

    def get_func(self, source, dest):
        return viscosity.MonaghanArtificialVsicosity(source=source, dest=dest, 
                                           alpha=self.alpha, beta=self.beta,
                                           gamma=self.gamma, eta=self.eta, 
                                           c=self.c)

class MorrisViscosity(Function):
    def __init__(self, mu='mu'):
        self.mu=mu
        
    def get_func(self, source, dest):
        return viscosity.MorrisViscosity(source=source, dest=dest,
                                         mu=self.mu)


class XSPHCorrection(Function):
    def __init__(self, eps=0.5):
        self.eps = eps
        
    def get_func(self, source, dest):
        return xsph.XSPHCorrection(source=source, dest=dest, eps=self.eps)


class XSPHDensityRate(Function):
    
    def get_func(self, source, dest):
        return xsph.XSPHDensityRate(source=source, dest=dest)
