"""
Classes for generators of some simple elements.
"""

# standard imports
import logging
logger = logging.getLogger()
import numpy

# local imports
from pysph.base.carray import DoubleArray, LongArray
from pysph.base.nnps import *
from pysph.base.point import Point

from pysph.solver.particle_generator import *
from pysph.solver.particle_generator import MassComputationMode as MCM
from pysph.solver.particle_generator import DensityComputationMode as DCM


###############################################################################
# `compute_particle_mass` function.
###############################################################################
def compute_particle_mass(parray, kernel, density=1000.0, h=0.1, dim=3):
    """
    Given a particle array, kernel, target density and interaction radius, find
    the mass of each particle.

    Note that this method works only when the particle radius is constant. This
    may also compute incorrect values when the particle cofiguration has voids
    within.
    """
    centroid = Point(0, 0, 0)
    dist = DoubleArray(0)
    indices = LongArray(0)
    
    x = parray.get('x')
    centroid.x = numpy.sum(x)/float(len(x))
    y = None
    z = None

    logger.debug('particles to compute_particle_mass %d'%(len(x)))
    
    if dim > 1:
        y = parray.get('y')
        centroid.y = numpy.sum(y)/float(len(y))
        if dim > 2:
            z = parray.get('z')
            centroid.z = numpy.sum(z)/float(len(z))
        else:
            z = numpy.zeros(len(x), dtype=numpy.float)
    else:
        y = numpy.zeros(len(x), dtype=numpy.float)
        z = y

    logger.debug('Centroid : %s'%(centroid))
    radius = kernel.radius()
    
    # find the nearest points in parray of the centroid.
    brute_force_nnps(pnt=centroid, search_radius=h*radius, 
                     xa=x, ya=y, za=z, 
                     neighbor_indices=indices,
                     neighbor_distances=dist)
    
    k = 0.0
    logger.info('Number of neighbors : %d'%(indices.length))
    pnt = Point()
    for i in range(indices.length):
        pnt.x = x[indices[i]]
        pnt.y = y[indices[i]]
        pnt.z = z[indices[i]]
        
        k += kernel.py_function(centroid, pnt, h)

    logger.info('Kernel sum : %f'%(k))
    logger.info('Requested density : %f'%(density))
    m = float(density/k)
    logger.info('Computed mass : %f'%(m))
    return m

###############################################################################
# `find_best_particle_spacing' function.
###############################################################################
def find_best_particle_spacing(length=1.0,
                               initial_spacing=0.1,
                               end_points_exact=True,
                               tolerance=1e-09):
    """
    Given the length and initial_spacing return a (possibly) corrected
    particle spacing and the number of points.
    """
    if length <= tolerance:
        return initial_spacing, 0

    n_intervals = int(numpy.floor(length/initial_spacing))
    if end_points_exact:
        r = length - n_intervals*initial_spacing
        new_spacing = initial_spacing + float(r/n_intervals)
    else:
        new_spacing = initial_spacing
        r = length - n_intervals*initial_spacing
        if r > numpy.fabs(length - ((n_intervals+1)*initial_spacing)):
            n_intervals += 1

    return new_spacing, (n_intervals+1)

###############################################################################
# `LineGenerator` class.
###############################################################################
class LineGenerator(ParticleGenerator):
    """
    Generate a line of points.
    """
    def __init__(self,
                 output_particle_arrays=[],
                 particle_mass=-1.0,
                 mass_computation_mode=MCM.Compute_From_Density,
                 particle_density=1000.0,
                 density_computation_mode=DCM.Set_Constant,
                 particle_h=0.1,
                 kernel=None,
                 start_point=Point(0, 0, 0),
                 end_point=Point(0, 0, 1),
                 particle_spacing=0.05,
                 end_points_exact=True,
                 tolerance=1e-09,
                 *args, **kwargs):
        """
        """
        self.start_point = Point(start_point.x,
                                 start_point.y,
                                 start_point.z)

        self.end_point = Point(end_point.x,
                                end_point.y,
                                end_point.z)
        
        self.particle_spacing = particle_spacing
        self.end_points_exact = end_points_exact
        self.tolerance = tolerance                 

    def get_coords(self):
        """
        Returns 3 numpy arrays representing the coordinates of the generated
        points.
        """
        
        dir = self.end_point - self.start_point
        distance = dir.length()
        
        if distance <= self.tolerance:
            x = numpy.asarray([], dtype=float)
            y = numpy.asarray([], dtype=float)
            z = numpy.asarray([], dtype=float)
            return x, y, z

        normal = dir/distance

        new_spacing, np = find_best_particle_spacing(
            length=distance,
            initial_spacing=self.particle_spacing,
            end_points_exact=self.end_points_exact,
            tolerance=self.tolerance)
        
        x = numpy.zeros(np, dtype=float)
        y = numpy.zeros(np, dtype=float)
        z = numpy.zeros(np, dtype=float)

        for i in range(np):
            x[i] = self.start_point.x + i*normal.x*new_spacing
            y[i] = self.start_point.y + i*normal.y*new_spacing
            z[i] = self.start_point.z + i*normal.z*new_spacing

        return x, y, z

    def validate_setup(self):
        """
        """
        return ParticleGenerator.validate_setup(self)

    def generate_func(self):
        """
        Generate a complete particle array with the required properties
        computed.
        """
        # setup the output particle array as required.
        self._setup_outputs()
        # find the coordinates
        x, y, z = self.get_coords()

        # add the generated particles to the output particle array
        output = self.output_particle_arrays[0]
        output.add_particles(x=x, y=y, z=z)
            
        # check if 'h' has to be set.
        if self.particle_h > 0.:
            output.h[:] = self.particle_h

        # check if density has to be set.
        if self.density_computation_mode == DCM.Set_Constant:
            output.rho[:] = self.particle_density
        
        # check if mass has to be set.
        if self.mass_computation_mode == MCM.Set_Constant:
            output.m[:] = self.particle_mass
        elif self.mass_computation_mode == MCM.Compute_From_Density:
            m = compute_particle_mass(density=self.particle_density, 
                                      h=self.particle_h, 
                                      parray=output, 
                                      kernel=self.kernel,
                                      dim=3)
            output.m[:] = m            

    def num_output_arrays(self):
        """
        Return the number of output particles arrays this generator will be
        generating.
        """
        return 1

###############################################################################
# `RectangleGenerator` class.
###############################################################################
class RectangleGenerator(ParticleGenerator):
    """
    Class to generate rectangles of particles - filled and hollow.
    """
    def __init__(self,
                 input_particle_arrays=[],
                 particle_mass=-1.0,
                 mass_computation_mode=MCM.Compute_From_Density,
                 particle_density=1000.0,
                 density_computation_mode=DCM.Set_Constant,
                 particle_h=0.1,
                 kernel=None,
                 filled=True,
                 start_point=Point(0, 0, 0),
                 end_point=Point(1, 1, 0),
                 particle_spacing_x1=0.1,
                 particle_spacing_x2=0.1,
                 end_points_exact=True,
                 tolerance=1e-09,
                 *args, **kwargs):
        """
        """
        ParticleGenerator.__init__(self, 
                                   input_particle_arrays=input_particle_arrays,
                                   particle_mass=particle_mass,
                                   mass_computation_mode=mass_computation_mode,
                                   particle_density=particle_density,
                                   density_computation_mode=density_computation_mode,
                                   particle_h=particle_h,
                                   kernel=kernel)

        self.filled = filled
        self.start_point = Point(start_point.x, start_point.y, start_point.z)
        self.end_point = Point(end_point.x, end_point.y, end_point.z)
        
        self.particle_spacing_x1 = particle_spacing_x1
        self.particle_spacing_x2 = particle_spacing_x2
        
        self.end_points_exact = end_points_exact
        self.tolerance = tolerance

    def num_output_arrays(self):
        """
        """
        return 1

    def validate_setup(self):
        """
        Make sure the input is valid.
        """
        if ParticleGenerator.validate_setup(self) == False:
            return False

        return self._validate_input_points()

    def _validate_input_points(self):
        """
        Make sure a proper rectangle has been requested by the input points.
        """
        dir = [0, 0, 0]
        
        if self.start_point.x != self.end_point.x:
            dir[0] = 1
        if self.start_point.y != self.end_point.y:
            dir[1] = 1
        if self.start_point.z != self.end_point.z:
            dir[2] = 1

        if sum(dir) != 2:
            msg = 'Incorrect input points specified'
            msg += '\n'
            msg += str(self.start_point)+' ,  '+str(self.end_point)
            logger.error(msg)
            return False

        return True

    def get_coords(self):
        """
        """
        # based on the input points, decide which is the plane this rectangle is
        # going to lie on.
        if self._validate_input_points() is False:
            return None

        dir = [0, 0, 0]
        
        if self.start_point.x != self.end_point.x:
            dir[0] = 1
        if self.start_point.y != self.end_point.y:
            dir[1] = 1
        if self.start_point.z != self.end_point.z:
            dir[2] = 1

        if dir[0] == 1:
            if dir[1] == 1:
                x, y, z = self._generate_x_y_rectangle()
            else:
                x, y, z = self._generate_x_z_rectangle()
        else:
            x, y, z = self._generate_y_z_rectangle()

        return x, y, z

    def _generate_x_y_rectangle(self):
        """
        Generate a rectangle in the x-y plane.
        """
        if self.start_point.x < self.end_point.x:
            start_x1 = self.start_point.x
            end_x1 = self.end_point.x
        else:
            start_x1 = self.end_point.x
            end_x1 = self.start_point.x

        if self.start_point.y < self.end_point.y:
            start_x2 = self.start_point.y
            end_x2 = self.end_point.y
        else:
            start_x2 = self.end_point.y
            end_x2 = self.start_point.y

        spacing1 = self.particle_spacing_x1
        spacing2 = self.particle_spacing_x2

        x, y = self.generate_rectangle_coords(start_x1=start_x1,
                                              start_x2=start_x2,
                                              end_x1=end_x1,
                                              end_x2=end_x2,
                                              spacing1=spacing1,
                                              spacing2=spacing2)

        z = numpy.zeros(len(x))
        
        return x, y, z

    def _generate_x_z_rectangle(self):
        """
        Generate a rectangle in the x-z plane.
        """
        if self.start_point.x < self.end_point.x:
            start_x1 = self.start_point.x
            end_x1 = self.end_point.x
        else:
            start_x1 = self.end_point.x
            end_x1 = self.start_point.x

        if self.start_point.z < self.end_point.z:
            start_x2 = self.start_point.z
            end_x2 = self.end_point.z
        else:
            start_x2 = self.end_point.z
            end_x2 = self.start_point.z

        spacing1 = self.particle_spacing_x1
        spacing2 = self.particle_spacing_x2

        x, z = self.generate_rectangle_coords(start_x1=start_x1,
                                              start_x2=start_x2,
                                              end_x1=end_x1,
                                              end_x2=end_x2,
                                              spacing1=spacing1,
                                              spacing2=spacing2)

        y = numpy.zeros(len(x))
        
        return x, y, z

    def _generate_y_z_rectangle(self):
        """
        Generate a rectangle in the y-z plane.
        """
        if self.start_point.y < self.end_point.y:
            start_x1 = self.start_point.y
            end_x1 = self.end_point.y
        else:
            start_x1 = self.end_point.y
            end_x1 = self.start_point.y

        if self.start_point.z < self.end_point.z:
            start_x2 = self.start_point.z
            end_x2 = self.end_point.z
        else:
            start_x2 = self.end_point.z
            end_x2 = self.start_point.z

        spacing1 = self.particle_spacing_x1
        spacing2 = self.particle_spacing_x2

        y, z = self.generate_rectangle_coords(start_x1=start_x1,
                                              start_x2=start_x2,
                                              end_x1=end_x1,
                                              end_x2=end_x2,
                                              spacing1=spacing1,
                                              spacing2=spacing2)

        x = numpy.zeros(len(y))
        
        return x, y, z

    def generate_rectangle_coords(self, start_x1, start_x2, end_x1, end_x2,
                                  spacing1, spacing2):

        """
        Generates a rectangle from the given start and end points, with the
        given spacing.
        """
        width = end_x1-start_x1
        height = end_x2-start_x2
        
        if width <= 0.0 or height <= 0.0 or spacing1 <= 0.0 or spacing2 <= 0:
            msg = 'Incorrect values :\n'
            msg = 'width=%f, height=%f, spacing1=%f, spacing1=%f'%(
                width, height, spacing1, spacing2)
            raise ValueError, msg

        new_spacing1, n1 = find_best_particle_spacing(length=width,
                                                      initial_spacing=spacing1,
                                                      end_points_exact=\
                                                          self.end_points_exact,
                                                      tolerance=self.tolerance)
        
        new_spacing2, n2 = find_best_particle_spacing(length=height,
                                                      initial_spacing=spacing2,
                                                      end_points_exact=\
                                                          self.end_points_exact,
                                                      tolerance=self.tolerance)

        if self.filled == False:
            n2 -= 2
            n = 2*n1 + 2*n2
        else:
            n = n1*n2

        x1 = numpy.zeros(n, dtype=float)
        x2 = numpy.zeros(n, dtype=float)

        if self.filled is True:
            pindx = 0
            for i in range(n1):
                for j in range(n2):
                    x1[pindx] = start_x1 + i*new_spacing1
                    x2[pindx] = start_x2 + j*new_spacing2
                    pindx += 1
        else:
            pindx = 0
            
            # generate the bottom horizontal lines
            for i in range(n1):
                x1[pindx] = start_x1 + i*new_spacing1
                x2[pindx] = start_x2
                pindx += 1

            end_x1 = x1[pindx-1]
        
            # now generate the left vertical line
            for i in range(n2):
                x1[pindx] = start_x1
                x2[pindx] = start_x2 + (i+1)*new_spacing2
                pindx += 1

            end_x2 = x2[pindx-1] + new_spacing2

            # the top 
            for i in range(n1):
                x1[pindx] = start_x1 + i*new_spacing1
                x2[pindx] = end_x2
                pindx += 1

            # the right side
            for i in range(n2):
                x1[pindx] = end_x1
                x2[pindx] = start_x2 + (i+1)*new_spacing2
                pindx += 1

        return x1, x2            

    def generate_func(self):
        """
        Generate a complete particle array with the required properties
        computed.
        """
        # setup the output particle array as required.
        self._setup_outputs()
        # find the coordinates
        x, y, z = self.get_coords()

        # add the generated particles to the output particle array
        output = self.output_particle_arrays[0]
        output.add_particles(x=x, y=y, z=z)
            
        # check if 'h' has to be set.
        if self.particle_h > 0.:
            output.h[:] = self.particle_h

        # check if density has to be set.
        if self.density_computation_mode == DCM.Set_Constant:
            output.rho[:] = self.particle_density
        
        # check if mass has to be set.
        if self.mass_computation_mode == MCM.Set_Constant:
            output.m[:] = self.particle_mass
        elif self.mass_computation_mode == MCM.Compute_From_Density:
            m = compute_particle_mass(density=self.particle_density, 
                                      h=self.particle_h, 
                                      parray=output, 
                                      kernel=self.kernel,
                                      dim=3)
            output.m[:] = m


###############################################################################
# `CuboidGenerator` class.
###############################################################################
class CuboidGenerator(ParticleGenerator):
    """
    Class to generate cuboids of particles (filled and hollow).
    """
    def __init__(self, 
                 output_particle_arrays=[],
                 particle_mass=-1.0,
                 mass_computation_mode=MCM.Compute_From_Density,
                 particle_density=1000.0,
                 density_computation_mode=DCM.Set_Constant,
                 particle_h=0.1,
                 kernel=None,
                 filled=True,
                 exclude_top=False,
                 start_point=Point(0, 0, 0),
                 end_point=Point(1, 1, 1),
                 particle_spacing_x=0.1,
                 particle_spacing_y=0.1,
                 particle_spacing_z=0.1,
                 end_points_exact=True,
                 tolerance=1e-09,
                 *args,
                 **kwargs):
        """
        Constructor.
        """
        self.filled = filled
        self.exclude_top = exclude_top
        self.start_point = Point(start_point.x,
                                 start_point.y,
                                 start_point.z)
        self.end_point = Point(end_point.x,
                                end_point.y,
                                end_point.z)
        
        self.particle_spacing_x = particle_spacing_x
        self.particle_spacing_y = particle_spacing_y
        self.particle_spacing_z = particle_spacing_z

        self.end_points_exact = end_points_exact
        self.tolerance = tolerance
        
    def num_output_arrays(self):
        """
        Number of particle arrays generated by this generated.
        """
        return 1

    def validate_setup(self):
        """
        Make sure the input is valid.
        """
        if ParticleGenerator.validate_setup(self) == False:
            return False
        
        return self._validate_input_points()

    def _validate_input_points(self):
        """
        Make sure the end points input are proper.
        """
        length = self.end_point.x - self.start_point.x
        depth = self.end_point.y - self.start_point.y
        width = self.end_point.z - self.start_point.z

        if (self.particle_spacing_x < 0.0 or
            self.particle_spacing_y < 0.0 or
            self.particle_spacing_z < 0.0 or
            abs(length)-self.particle_spacing_x < 0. or
            abs(depth)-self.particle_spacing_y < 0. or
            abs(width)-self.particle_spacing_z < 0.):
            msg = 'Incorrect input paramters specified'
            logger.error(msg)
            return False

        return True
            
    def get_coords(self):
        """
        Returns 3 numpy arrays representing the coordinates of the generated points.
        """
        start_point, end_point, length, depth, width = self._get_end_points()
        
        particle_spacing_x, nx = find_best_particle_spacing(
            length=length, 
            initial_spacing=self.particle_spacing_x)
        particle_spacing_y, ny = find_best_particle_spacing(
            length=depth,
            initial_spacing=self.particle_spacing_y)
        particle_spacing_z, nz = find_best_particle_spacing(
            length=width,
            initial_spacing=self.particle_spacing_z)

        logger.info('x-spacing : %f, nx : %d'\
                        %(particle_spacing_x, nx))
        logger.info('y-spacing : %f, ny : %d'\
                        %(particle_spacing_y, ny))
        logger.info('z-spacing : %f, nz : %d'\
                        %(particle_spacing_z, nz))

        if self.filled == True:
            return  self._generate_filled_cuboid(start_point, end_point, 
                                                 nx, ny, nz,
                                                 particle_spacing_x, 
                                                 particle_spacing_y,
                                                 particle_spacing_z)
        else:
            return self._generate_empty_cuboid(start_point, end_point,
                                               nx, ny, nz,
                                               particle_spacing_x,
                                               particle_spacing_y,
                                               particle_spacing_z)        
    def generate_func(self):
        """
        Generate the complete particle array with the required properties computed.
        """
        self._setup_outputs()

        # compute the coords.
        x, y, z= self.get_coords()
        
        # add the generated particles to the output particle array
        output = self.output_particle_arrays[0]
        output.add_particles(x=x, y=y, z=z)
        
        # check if 'h' has to be set
        if self.particle_h > 0.:
            output.h[:] = self.particle_h

        # check if the density is to be set.
        if self.density_computation_mode == DCM.Set_Constant:
            output.rho[:] = self.particle_density
        
        # check if the mass has to be computed.
        if self.mass_computation_mode == MCM.Set_Constant:
            output.m[:] = self.particle_mass
        elif self.mass_computation_mode == MCM.Compute_From_Density:
            m = compute_particle_mass(density=self.particle_density,
                                      h=self.particle_h,
                                      parray=output,
                                      kernel=self.kernel,
                                      dim=3)
            output.m[:] = m
        
    def _get_end_points(self):
        """
        Return changed end points so that start_point to end_point is moving in
        positive direction for all coords.
        """
        length = self.end_point.x - self.start_point.x
        depth = self.end_point.y - self.start_point.y
        width = self.end_point.z - self.start_point.z

        start_point = Point()
        end_point = Point()

        if length < 0:
            start_point.x = self.end_point.x
            end_point.x = self.start_point.x
        else:
            start_point.x = self.start_point.x
            end_point.x = self.end_point.x

        if depth < 0:
            start_point.y = self.end_point.y
            end_point.y = self.start_point.y
        else:
            start_point.y = self.start_point.y
            end_point.y = self.end_point.y

        if width < 0:
            start_point.z = self.end_point.z
            end_point.z = self.start_point.z
        else:
            start_point.z = self.start_point.z
            end_point.z = self.end_point.z
            
        return start_point, end_point, abs(length), abs(depth), abs(width)

    def _generate_empty_cuboid(self, start_point, end_point, nx, ny, nz,
                               particle_spacing_x, particle_spacing_y,
                               particle_spacing_z):
        """
        """
        logger.info('Input num pts : %d %d %d'%(nx, ny, nz))

        if self.exclude_top is True:
            ny -= 1

        n = 0
        n += 2*nx*ny # for the z-max and z-min planes
        nz -= 2
        
        if self.exclude_top is False:
            n += 2*nx*nz # for the y-max and y-min planes
            ny -= 2
        else:
            n += nx*nz
            ny -= 1        

        n += 2*ny*nz # for the x-max and x-min planes

        if self.exclude_top is False:
            ny += 2
        else:
            ny += 1

        nz += 2
        
        x = numpy.zeros(n, dtype=float)
        y = numpy.zeros(n, dtype=float)
        z = numpy.zeros(n, dtype=float)

        pindx = 0

        logger.info('Now using num pts : %d %d %d'%(nx, ny, nz))
        logger.info('Computed number of points : %d'%(n))
        
        # generate the z-min and max planes
        for i in range(nx):
            for j in range(ny):
                x[pindx] = start_point.x + i*particle_spacing_x
                y[pindx] = start_point.y + j*particle_spacing_y
                z[pindx] = start_point.z
                pindx += 1
        for i in range(nx):
            for j in range(ny):
                x[pindx] = start_point.x + i*particle_spacing_x
                y[pindx] = start_point.y + j*particle_spacing_y
                z[pindx] = end_point.z
                pindx += 1

        # generate the bottom and top planes
        for i in range(nx):
            for k in range(nz-2):
                x[pindx] = start_point.x + i*particle_spacing_x
                y[pindx] = start_point.y
                z[pindx] = start_point.z + (k+1)*particle_spacing_z
                pindx += 1
        if self.exclude_top is False:
            for i in range(nx):
                for k in range(nz-2):
                    x[pindx] = start_point.x + i*particle_spacing_x
                    y[pindx] = end_point.y
                    z[pindx] = start_point.z + (k+1)*particle_spacing_z
                    pindx += 1

        # generate the left and right planes
        if self.exclude_top is True:
            ny += 1

        for j in range(ny-2):
            for k in range(nz-2):
                x[pindx] = start_point.x
                y[pindx] = start_point.y + (j+1)*particle_spacing_y
                z[pindx] = start_point.z + (k+1)*particle_spacing_z
                pindx += 1
        for j in range(ny-2):
            for k in range(nz-2):
                x[pindx] = end_point.x
                y[pindx] = start_point.y + (j+1)*particle_spacing_y
                z[pindx] = start_point.z + (k+1)*particle_spacing_z
                pindx += 1

        logger.info('Last pindx value : %d'%(pindx))
        return x, y, z

    def _generate_filled_cuboid(self, start_point, end_point, nx, ny, nz,
                                particle_spacing_x, particle_spacing_y,
                                particle_spacing_z):
        """
        """
        n = nx*ny*nz

        x = numpy.zeros(n, dtype=float)
        y = numpy.zeros(n, dtype=float)
        z = numpy.zeros(n, dtype=float)
        
        pindx = 0

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x[pindx] = start_point.x + i*particle_spacing_x
                    y[pindx] = start_point.y + j*particle_spacing_y
                    z[pindx] = start_point.z + k*particle_spacing_z
                    pindx += 1

        return x, y, z

