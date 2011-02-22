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
import funcs.adke_funcs as adke

class Function(object):
    """ Base class that defines an sph function (sph.funcs) and it's
    associated parameter values.

    Methods:
    --------

    get_func -- Return a particular instance of SPHFunctionParticle
    with an appropriate source and destincation particle araay.

    Example:
    --------

    The sph function MonaghanArtificialVsicosity (defined in
    sph.funcs.viscosity_funcs) requires the parameter values 'alpha',
    'beta', 'gamma' and 'eta' to define the artificial viscosity. This
    function may be created as:

    avisc = MonaghanArtificialVsicosity(hks=False, alpha, beta, gamma, eta)
    avisc_func = avisc.get_funcs(source, dest)

    Thus, Function provides us a means to create functions at will
    between source and destination particle arrays with specified
    parameter values.

    """
    def __init__(self, sph_func=None, hks=False, *args, **kwargs):
        """ Base class Constructor

        Parameters:
        -----------

        sph_func -- the SPHFunctionParticle class type. Defaults to None
        *args -- optional positional arguments.
        **kwargs -- optional keyword arguments.

        Notes:
        ------

        """
        self.sph_func = sph_func
        self.args = args
        self.kwargs = kwargs
        self.hks = hks

    def get_func(self, source, dest):
        """ Return a SPHFunctionParticle instance with source and dest """
        if self.sph_func is None:
            raise NotImplementedError, 'Function(sph_func=None).get_func()'

        func = self.sph_func(source, dest, hks=self.hks,
                             *self.args, **self.kwargs)
        
        return func

############################################################################
# SPHInterpolation
############################################################################
class SPHInterpolation(Function):
    """ This function interpolates a scalar property sampled at the location
    of source onto the location of the destination.

    The formula for interpolation is

    ..math::

    <f>_a = \sum_{b=1}^{b=N} \frac{m_b}{\rho_b} f_b W_{ab}

    The function used is SPH defined in sph.funcs.basic_funcs

    """
    
    def __init__(self, prop_name = "", hks = False):
        """ Constructor.

        Parameters:
        -----------

        prop_name -- the property to interpolate
        hks -- Hernquist Katz kernel symmetrization. Defaults to False

        """
        Function.__init__(self, basic.SPH, prop_name=prop_name, hks = hks)

############################################################################
# SPHSimpleGradient
############################################################################    
class SPHSimpleGradient(Function):
    """ Compute the gradient of a scalar property sampled at the
    location of source at the location of destination.

    The formula used is

    ..math::

    <\nabla f>_a = \sum_{b=1}^{b=N} \frac{m_b}{\rho_b}f_b \nabla_a W_{ab}

    The function used is SPHGradient defined in sph.funcs.basic_funcs

    """
    
    def __init__(self, prop_name="", hks = False):
        """ Constructor.

        Parameters:
        -----------

        prop_name -- the property for which the gradient is required.
        hks -- Hernquist Katz kernel symmetrization. Defaults to False

        """        
        Function.__init__(self, basic.SPHGradient, prop_name=prop_name,
                          hks = hks)

############################################################################
# SPHGradient
############################################################################
class SPHGradient(Function):
    """ Compute the gradient of a scalar property sampled at the
    location of source at the location of destination.

    The formula used is

    ..math::

    <\nabla f>_a = \sum_{b=1}^{b=N} \frac{m_b}{\rho_b}(f_b - f_a)\nabla_a W_{ab}

    The function used is SPHGradient defined in sph.funcs.basic_funcs

    """
    
    def __init__(self, prop_name="", hks = False):
        Function.__init__(self, basic.SPHGradient, prop_name=prop_name, hks=hks)

############################################################################
# SPHLaplacian
############################################################################
class Laplacian(Function):
    """ Compute the laplacian of a scalar property sampled at the
    location of source at the location of destination.

    The formula used is

    ..math::

    The function used is SPHLaplacian defined in sph.funcs.basic_funcs

    """        
   
    def __init__(self, prop_name="", hks=False):
        Function.__init__(self, basic.SPHLaplacian, prop_name=prop_name,
                          hks=hks)
        
############################################################################
# MonaghanBoundaryForce
############################################################################
class MonaghanBoundaryForce(Function):
    """ Compute the boundary force using the formula

    ..math::

    \vec{f} = \frac{m_b}{m_a + m_b} B(x,y)\vec{n_b}

    B(x,y) = \Gamma (y) \Xi (x)

    This is defined in "Smoothed Particle Hydrodynamics" by J.J. Monaghan 2005

    The function used id MonaghanBoundaryForce defined in
    sph.funcs.boundary_funcs

    """   
    
    def __init__(self, delp = 1.0):
        Function.__init__(self, boundary.MonaghanBoundaryForce, delp = delp)

        
############################################################################
# BeckerBoundaryForce
############################################################################
class BeckerBoundaryForce(Function):
    """ Compute the boundary force using the formula described in becker07
    
    ::math::
    
    f_ak = \frac{m_k}{m_a + m_k}\tau{x_a,x_k}\frac{x_a -
    x_k}{\lvert x_a - x_k \rvert} where \tau{x_a, x_k} =
    0.02\frac{c_s^2}{\lvert x_a - x_k|\rvert}\begin{cases}
    2/3 & 0 < q < 2/3\\ (2q - 3/2q^2) & 2/3 < q < 1\\
    1/2(2-q)^2 & 1 < q < 2\\ 0 & \text{otherwise}
    \end{cases}
    
    The function used is BeckerBoundaryForce defined in
    sph.funcs.boundary_funcs

    """    
    
    def __init__(self, sound_speed):
        Function.__init__(self, boundary.BeckerBoundaryForce,
                          sound_speed=sound_speed)
############################################################################
# LennardJonesForce
############################################################################
class LennardJonesForce(Function):
    """Compute the boundary force using the Lennard Jones Potential

    The foumula used is

    ..math::

    The function used is LennardJonesForce defined in sph.funcs.boundary_funcs

    """
    
    def __init__(self, D, ro, p1, p2):
        Function.__init__(self, boundary.LennardJonesForce,
                          D=D, ro=ro, p1=p1, p2=p2)

############################################################################
# SPHRho
############################################################################
class SPHRho(Function):
    """ SPH Summation density

    ..math::

    <\rho>_a = m_b W_{ab}

    The function used is density.SPHRho defined in sph.funcs.density_funcs

    """
    def __init__(self, hks=False):
        Function.__init__(self, density.SPHRho, hks=hks)

############################################################################
# SPHDensityRate
############################################################################
class SPHDensityRate(Function):
    """ Continuity equation  

    ..math::

    \frac{D\rho_a}{Dt} = m_b (v_a - v_b)\,\nabla_a W_{ab}

    The function used is density.SPHDensityRate defined in
    sph.funcs.density_funcs

    """
    def __init__(self, hks=False):
        Function.__init__(self, density.SPHDensityRate, hks=hks)

############################################################################
# EnergyEquationNoVisc
############################################################################
class EnergyEquationNoVisc(Function):
    """ Standard thermal energy equation without artificial viscosity

    ..math::

    The function used is EnergyEquationNoVisc

    """
    def __init__(self, hks=False):
        Function.__init__(self, energy.EnergyEquationNoVisc, hks=hks)

############################################################################
# EnergyEquationAVisc
############################################################################
class EnergyEquationAVisc(Function):
    """ Monaghan artificial viscosity for the thermal energy equation

    ..math::

    The function used is EnergyEquationAVisc

    """    
    def __init__(self, beta=1.0, alpha=1.0, gamma=1.4, eta=0.1, hks=False):
        Function.__init__(self, energy.EnergyEquationAVisc, alpha=alpha,
                          beta=beta, gamma=gamma, eta=eta, hks=hks)

############################################################################
# EnergyEquation
############################################################################
class EnergyEquation(Function):
    """ Standard thermal energy equation with artificial viscosity

    ..math::

    The function used is EnergyEquation defined in sph.funcs.energy_funcs

    """    
    def __init__(self, hks=False, beta=1.0, alpha=1.0, gamma=1.4, eta=0.1):
        Function.__init__(self, energy.EnergyEquation, alpha=alpha,
                          beta=beta, gamma=gamma, eta=eta, hks=hks)

############################################################################
# ArtificialHeat
############################################################################
class ArtificialHeat(Function):
    """ Artificial heat condiction term

    ..math::

    \frac{1}{\rho}\nabla\,\cdot(q\nabla(u)) = -\sum_{b=1}^{b=N}
    m_b \frac{(q_a + q_b)(u_a - u_b)}{\rho_{ab}(|\vec{x_a} - \vec{x_b}|^2 +
    (h\eta)^2)}\,(\vec{x_a} - vec{x_b})\cdot \nabla_a W_{ab}

    q_a = h_a (g1 c_a + g2 h_a (abs(div_a) - div_a))

    The function used is ArtificialHeat defined in sph.funcs.energy_funcs

    """    
    def __init__(self, g1=0.02, g2=0.4, eta=0.1, hks=False):
        Function.__init__(self, energy.ArtificialHeat, g1=g1, g2=g2,
                          hks=hks)

############################################################################
# IdealGasEquation
############################################################################
class IdealGasEquation(Function):
    """ Ideal gas EOS

    ..math::

    p = (\gamma - 1) \rho e

    The function used is IdealGasEquation defined in sph.funcs.eos_funcs

    """
    def __init__(self, gamma=1.4):
        Function.__init__(self, eos.IdealGasEquation, gamma=gamma)

############################################################################
# TaitEquation
############################################################################
class TaitEquation(Function):
    """ Tait eos for nearly incompressible fluids

    ..math::

     p = B((\frac{\rho}{\rho_0})^{\gamma} - 1)

     The function used is TaitEquation defined in sph.funcs.eos_funcs

    """
    
    def __init__(self, co, ro, gamma=7.0):
        Function.__init__(self, eos.TaitEquation, co=co, ro=ro,
                          gamma=gamma)

############################################################################
# PositionStepping
############################################################################
class PositionStepping(Function):
    """ Position stepping

    ..math::

    \frac{Dx_a}{Dt} = v_a

    """
    def __init__(self):
        Function.__init__(self, position.PositionStepping)

############################################################################
# GravityForce
############################################################################
class GravityForce(Function):
    """ Simple gravity force

    ..math::

    \frac{Dv_a}{Dt} = \vec{g}

    """
    def __init__(self, gx=0.0, gy=-9.81, gz=0.0):
        Function.__init__(self, external.GravityForce, gx=gx,gy=gy,gz=gz)    

############################################################################
# SPHPressureGradient
############################################################################
class SPHPressureGradient(Function):
    """ Momentum equation pressure gradient term

    ../math::

    \frac{Dv_a}{Dt} = -m_b (\frac{p_a}{\rho_a^2} +
    \frac{p_b}{\rho_b^2})\nabla_a W_{ab}

    the function used is SPHPressureGradient defined in sph.funcs.pressure_funcs

    """

    def __init__(self, hks=False):
        Function.__init__(self, pressure.SPHPressureGradient, hks=hks)

############################################################################
# MomentumEquation
############################################################################    
class MomentumEquation(Function):
    """ Standard Momentum equation with artificial viscosity

    ../math::

    \frac{Dv_a}{Dt} = -m_b (\frac{p_a}{\rho_a^2} +
    \frac{p_b}{\rho_b^2} + \Pi_{ab})\nabla_a W_{ab}

    the function used is MomentumEquation defined in sph.funcs.pressure_funcs

    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.4, eta=0.1,
                 hks=False):
        Function.__init__(self, pressure.MomentumEquation, alpha=alpha,
                          beta=beta, gamma=gamma, eta=eta, hks=hks)

############################################################################
# MonaghanArtificialVsicosity
############################################################################    
class MonaghanArtificialVsicosity(Function):
    """ Monaghan type artificial viscosity

    ..math::


    the function used is MonaghanArtificialVsicosity defined in
    sph.funcs.viscosity_funcs

    """
    
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.4, eta=0.1, hks=False):
        Function.__init__(self, viscosity.MonaghanArtificialVsicosity,
                          alpha=alpha, beta=beta, gamma=gamma, eta=eta, hks=hks)

############################################################################
# MorrisViscosity
############################################################################    
class MorrisViscosity(Function):
    """ Morris type physical viscosity

    ..math::


    the function used is MorrisViscosity defined in sph.funcs.viscosity_funcs

    """

    def __init__(self, mu='mu', hks=False):
        Function.__init__(self, viscosity.MorrisViscosity, mu=mu, hks=hks)
        
############################################################################
# XSPHCorrection
############################################################################    
class XSPHCorrection(Function):
    """ Standard XSPH correction

    ..math::

    \del_a = \epsilon \frac{m_b (v_b - v_a}{frac{\rho_a + \rho_b}{2}} W_{ab}

    the function used is XSPHCorrection defined in sph.funcs.xsph_funcs

    """
    
    
    def __init__(self, eps=0.5, hks=False):
        Function.__init__(self, xsph.XSPHCorrection, eps=eps, hks=hks)

############################################################################
# ADKEPilotRho
############################################################################    
class ADKEPilotRho(Function):
    """ Pilot estimate for ADKE Smoothing

    ..math::

    \rho_a = m_b W(x_a-x_b, h0)

    """
    
    def __init__(self, h0=1.0, hks=False):
        Function.__init__(self, adke.PilotRho, h0=h0, hks=hks)

############################################################################
# VelocityDivergence
############################################################################    
class VelocityDivergence(Function):
    """ Divergence of Velocity

    ..math::

    (\nabla\,\cdot \vec{v}) = \frac{1}{\rho_a} m_b (v_b - v_a)\cdot
    \nabla_a W_{ab}

    """
    def __init__(self, hks=False):
        Function.__init__(self, adke.PilotRho, hks=hks)

############################################################################
# VectorForce
############################################################################    
class VectorForce(Function):
    """ Arbitrary vector force """
    def __init__(self, force):
        Function.__init__(self, external.VectorForce, force=force)

############################################################################
# MoveCircleX
############################################################################    
class MoveCircleX(Function):
    """ Test function for integration test """
    def __init__(self, hks=False):
        Function.__init__(self, external.MoveCircleX, hks=False)

############################################################################
# MoveCircleY
############################################################################    
class MoveCircleY(Function):
    """ Test function for integration test """ 
    def __init__(self, hks=False):
        Function.__init__(self, external.MoveCircleY, hks=False)

############################################################################
# XSPHDensityRate
############################################################################    
class XSPHDensityRate(Function):
    """ XSPHDensityRate """
    def get_func(self, source, dest):
        func = xsph.XSPHDensityRate(source=source, dest=dest)
        func.tag = "density"
        return func

############################################################################
# NeighborCount
############################################################################
class NeighborCount(Function):
    """ Compute the number of neighbors """
    def __init__(self):
        Function.__init__(self, basic.CountNeighbors)

class SPHFunction(Function):
    def __init__(self):
        Function.__init__(self, basic.SPHFunction)
