"""
Contains class to perform load balancing.
"""

# logging imports
import logging
logger = logging.getLogger()

# standard imports
import numpy

# local imports
from pysph.base.particle_array import ParticleArray


TAG_LB_PARTICLE_REQUEST = 101
TAG_LB_PARTICLE_REPLY = 102

###############################################################################
# `LoadBalancer` class.
###############################################################################
class LoadBalancer:
    """
    Class to perform simple load balancing.
    """
    def __init__(self, parallel_solver=None, parallel_cell_manager=None, *args, **kwargs):
        """
        Constructor.
        """
        self.setup_done = False
        self.cell_manager = parallel_cell_manager
        self.solver = parallel_solver
        self.skip_iteration = 10
        self.proc_map = None
        self.pid = 0
        self.num_procs = 1
        self.current_iteration = 0
        self.particles_per_proc = []
        self.ideal_load = 0.
        self.threshold_ratio = 25.
        self.threshold_margin = 0.
        self.lb_max_iterations = 10
        self.upper_threshold = 0.
        self.lower_threshold = 0.
        self.load_difference = []
        self.communicating_procs = []
        self.has_zero_procs = False
        self.prev_particle_count = []
        
    def setup(self):
        """
        Sets up some internal data.
        """
        if self.setup_done == True:
            return

        self.parallel_controller = self.cell_manager.parallel_controller
        self.pid = self.parallel_controller.rank
        self.proc_map = self.cell_manager.proc_map
        self.num_procs = self.parallel_controller.num_procs
        self.comm = self.parallel_controller.comm

        self.setup_done = True

    def load_balance(self):
        """
        The load balance function.
        """
        self.setup()
        
        if self.solver is None or self.solver.current_iteration % self.skip_iteration == 0:
            self.load_balance_func()

        self.current_iteration += 1

    def load_balance_func(self):
        """
        Perform the load balancing.
        
        **Algorithm**
        
            - while load not balanced or lb iterations not exceeded.

                - Compute some statistics
                - Find the number of real particles in all processors.
                - Find the total number of particles.
                - Find the mean number of particles with each processor.
                - If number of particles with each processor is within a
                  particular threshold from the mean, load is balacned, exit.
                
                - Sort processor ids in increasing order of number of particles
                  with them. In case of multiple processors having the same
                  number of particles, arrange them in ascending order of pid.
                
                - If there are some processors with 0 particles, communication
                  among all processors.
                - If no such processors are there, each processor shall
                  communicate with adjacent neighbors.
               
                - *********** PASS1 ************
                - mypid <- self.rank
                - num_procs <- len(procs_to_communicate)
                - i = num_procs-1
                - pid <- procs_to_communicate[i]
                - while pid != mypid:
                    - send request to pid for particles.
                    - recv particles of one or more cells from pid
                    - add particles to particle array.
                    - i -= 1
                
                - *********** PASS2 ************
                - i = 0
                - pid <- procs_to_communicate[i]
                - while pid != mypid:
                    - recv request from pid for particles.
                    - find a suitable set of cells to offload.
                    - send particles of these cells to pid.
                    - remove sent particles from local cells.
                    - i += 1

                - BARRIER.
                - bin particles top down.
                - update processor map.
                - update neighbor information.

                - lb_iterations += 1
        """
        balancing_done = False
        current_balance_iteration = 0
        num_procs = self.num_procs
        self.particles_per_proc = [0]*num_procs
        if len(self.prev_particle_count) == 0:
            self.prev_particle_count = [0]*num_procs
        self.ideal_load = 0.
        self.load_difference = [0]*num_procs
        
        while balancing_done == False:
            logger.info('Load Balance iteration %d -------------------'%(
                    current_balance_iteration))

            if current_balance_iteration >= self.lb_max_iterations:
                balancing_done = True
                logger.info('MAX LB ITERATIONS EXCEEDED')
                continue
            
            # get the number of particles with each process.
            self.particles_per_proc = self.collect_num_particles()
            self.total_particles = sum(self.particles_per_proc)
            self.ideal_load = float(self.total_particles)/num_procs
            
            self.threshold_margin = self.ideal_load*self.threshold_ratio/100.
            self.lower_threshold = self.ideal_load - self.threshold_margin
            self.upper_threshold = self.ideal_load + self.threshold_margin
            
            for i in range(num_procs):
                self.load_difference[i] = (self.particles_per_proc[i] - 
                                           self.ideal_load)

            min_diff = min(self.load_difference)
            max_diff = max(self.load_difference)
            
            if (abs(min_diff) < self.threshold_margin and max_diff <
                self.threshold_margin):
                balancing_done = True
                logger.info('BALANCE ACHIEVED')
                logger.debug('Num particles are : %s'%(self.particles_per_proc))
                continue

            if self.particles_per_proc == self.prev_particle_count:
                # meaning that the previous load balancing iteration did not
                # change the particle counts, we do not do anything now.
                balancing_done = True
                logger.info('Load unchanged')
                continue

            logger.debug('Total particles : %d'%(self.total_particles))
            logger.debug('Ideal load : %d'%(self.ideal_load))
            logger.debug('Load DIfference : %s'%(self.load_difference))
            logger.info('Particle counts : %s'%(self.particles_per_proc))
            logger.debug('Threshold margin: %f'%(self.threshold_margin))
            logger.debug('Upper threshold : %f'%(self.upper_threshold))
            logger.debug('Lower threshold : %f'%(self.lower_threshold))
            
            if min(self.particles_per_proc) == 0:
                self.load_balance_with_zero_procs()
            else:
                self.load_balance_normal()

            # update the cell information.
            self.cell_manager.remove_remote_particles()
            self.cell_manager.bin_particles_top_down()
            self.cell_manager.glb_update_proc_map()
            self.cell_manager.update_cell_neighbor_information()
            
            self.comm.Barrier()

            current_balance_iteration += 1
            # store the old particle counts in prev_particle_count
            self.prev_particle_count[:] = self.particles_per_proc

    def collect_num_particles(self):
        """
        Finds the number of particles with each processor. 

        **Algorithm**

            - gather each processors particle count at the root.
            - scatter this data to all processors.

        """
        arrays = self.cell_manager.arrays_to_bin
        num_particles = sum(map(ParticleArray.get_number_of_particles, arrays))

        particles_per_proc = self.comm.gather(num_particles, root=0)
        # now num_particles has one entry for each processor, containing the
        # number of particles with each processor. broadcast that data to all
        # processors.
        particles_per_proc = self.comm.bcast(particles_per_proc, root=0)
        return particles_per_proc
    
    def load_balance_normal(self):
        """
        The normal diffusion based load balance algorithm.
        """
        self.procs_to_communicate = self._get_procs_to_communicate(
            self.particles_per_proc,
            self.cell_manager.adjacent_processors)
        num_procs = len(self.procs_to_communicate)

        # PASS 1
        num_procs = len(self.procs_to_communicate)
        i = num_procs - 1

        pid = self.procs_to_communicate[i]
        while pid != self.pid:
            self.normal_lb_pass1(pid)
            i -= 1
            pid = self.procs_to_communicate[i]

        # PASS 2
        i = 0
        pid = self.procs_to_communicate[i]
        while pid != self.pid:
            self.normal_lb_pass2(pid)
            i += 1
            pid = self.procs_to_communicate[i]

    def load_balance_with_zero_procs(self):
        """
        Balances load when there are some processes with no particles.

        **Idea**

        If a process has zero particles, it requests the process with the
        highest number of particles(at the start of the algorithm) for
        particles. The process may or may not donate particles. If the zero
        particle proc gets particles from this process, it will send empty
        requests to the rest of the non-zero particle procs. Each zero particle
        proc does this until it finds the first process ready to donate
        particles.

        **Algorithm**

            - if process is zero particle proc, then starting with the proc
              having highest number of proc start requesting all other procs,
              till another zero particle proc is reached.
            - if process is non-zero particle proc, then starting with the first
              proc having zero particles, respond to requests from each proc.

        """
        num_procs = self.num_procs

        self.procs_to_communicate = self._get_procs_to_communicate(
            self.particles_per_proc, 
            list(numpy.arange(self.num_procs)))

        if self.particles_per_proc[self.pid] == 0:
            self._zero_request_particles()
        else:
            self._zero_donate_particles()

    def _get_procs_to_communicate(self, particles_per_proc, procs_to_communicate):
        """
        Returns the list of procs in correct order to communicate with during
        load balancing. The procs will be same as in the list
        procs_to_communicate but will be ordered properly in order to avoid any
        deadlocks. 

        The returned proc list will have process id's sorted in increasing order
        of the number of particles in them. In case of ties, lower process id
        will appear before a higher process id.
        
        **Parameters**

            - particles_per_proc - the number of particles with EVERY processor
              in the world.
            - procs_to_communicate - list of procs to communicate with while
              load balancing. 

        """
        proc_order = list(numpy.argsort(particles_per_proc, kind='mergesort'))

        for i in range(len(proc_order)-1):
            if particles_per_proc[proc_order[i]] ==\
                    particles_per_proc[proc_order[i+1]]:
                if proc_order[i] > proc_order[i+1]:
                    # swap the two
                    temp = proc_order[i]
                    proc_order[i] = proc_order[i+1]
                    proc_order[i+1] = temp
                    
        # select from this sorted order, the procs in procs_to_communicate.
        output_procs = []
        for proc in proc_order:
            if procs_to_communicate.count(proc) == 1:
                output_procs.append(proc)
        
        return output_procs
            
    def normal_lb_pass1(self, pid):
        """
        Requests for particles from processors having more particles than self.

        **Algorithm**
        
            - send request for particles to pid.
            - recv reply.
            - depending on reply add new particles to self.

        **Data sent/received**

            - check if we need more particles.
            - if yes
                - send a dictionary in the format given below.
                - receive a dictionary of cell with particles for them, the
                dictionary could be empty.
                - add particles received in the particle arrays as real
                particles. 
            - if no
                - send a dictionary in the format given below.
                - receive an empty dictionary.

        """
        logger.debug('Requesting %d for particles'%(pid))
        send_data = self._build_particle_request()

        self.comm.send(send_data, dest=pid, tag=TAG_LB_PARTICLE_REQUEST)
        data = self.comm.recv(source=pid, tag=TAG_LB_PARTICLE_REPLY)
        particle_data = data['particles']

        self.cell_manager.add_local_particles_to_parray(particle_data)
        
        logger.debug('DONE')

    def normal_lb_pass2(self, pid):
        """
        Process requests from processors with lesser particles than self.

        **Algorithm**

            - recv request from pid.
            - if pid requested particles
            - check if we have particles enough to give.
            - if yes, choose an appropriate cell(s), extract particles and send.

        """
        logger.debug('Processing request from %d'%(pid))
        comm = self.comm
        arrays = self.cell_manager.arrays_to_bin
        num_particles = sum(map(ParticleArray.get_number_of_particles, arrays))

        request = comm.recv(source=pid, tag=TAG_LB_PARTICLE_REQUEST)
        reply = self._build_particle_request_reply(request, pid)
        comm.send(reply, dest=pid, tag=TAG_LB_PARTICLE_REPLY)

        logger.debug('process request DONE')
        
    def _build_particle_request(self):
        """
        Build the dictionary to be sent as a particle request.
        """
        arrays = self.cell_manager.arrays_to_bin
        num_particles = sum(map(ParticleArray.get_number_of_particles, arrays))
        data = {}

        if num_particles < self.ideal_load:
            data['need_particles'] = True
            data['num_particles'] = num_particles
        else:
            data['need_particles'] = False

        return data

    def _build_particle_request_reply(self, request, pid):
        """
        Build the reply to be sent in response to a request.
        """
        arrays = self.cell_manager.arrays_to_bin
        num_particles = sum(map(ParticleArray.get_number_of_particles, arrays))

        reply = {}

        if request['need_particles'] == False:
            logger.debug('%d request for NO particles'%(pid))
            reply['particles'] = {}
            return reply

        num_particles_in_pid = request['num_particles']
        
        # check if pid has more particles than us.
        if num_particles_in_pid >= num_particles:
            logger.debug('%d has more particles that %d'%(pid, self.pid))
            reply['particles'] = {}
            return reply
        
        # if our number of particles is within the threshold, do not donate
        # particles. 
        if abs(self.ideal_load-num_particles) < self.threshold_margin:
            if (not (num_particles-num_particles_in_pid) >
                self.threshold_margin):
                logger.debug('Need not donate - not overloaded')
                reply['particles'] = {}
                return reply

        # if we have only one cell, do not donate.
        if len(self.cell_manager.cells_dict) == 1:
            logger.debug('Have only one cell - will not donate')
            reply['particles'] = {}
            return reply

        # get one or more cells to send to pid
        data = self._get_particles_for_neighbor_proc(pid)
        reply['particles'] = data
        return reply
    
    def _get_particles_for_neighbor_proc(self, pid):
        """
        Returns particles from one or more cells to be moved to pid for
        processing.

        **Algorithm**

            - Get all cells with self that have remote neighbors.
            - Of these get all particles that have 'pid' as neighbor.
            - Of these choose the one with the most amount of neighbor cells in
              pid.
            -
        """
        cells = self.cell_manager.cells_dict
        max_neighbor_count = -1
        cells_for_nbr = []

        for cid, c in cells.iteritems():
            if c.pid != self.pid:
                logger.debug('%s not local cell'%(cid))
                continue

            pinfo = c.parallel_cell_info

            logger.debug('%s\'s pinfo, %s'%(cid, pinfo.neighbor_cell_pids.values()))
            pinfo.compute_neighbor_counts()
            logger.debug('remote nbr cnts : %s'%(pinfo.remote_pid_cell_count))

            if pinfo.num_remote_neighbors == 0:
                #logger.debug('%s has no remote nbrs'%(cid))
                continue

            num_nbrs_in_pid = pinfo.remote_pid_cell_count.get(pid)
            
            if num_nbrs_in_pid is None:
                continue

            if num_nbrs_in_pid > max_neighbor_count:
                max_neighbor_count = num_nbrs_in_pid
                cells_for_nbr = [cid]
            elif num_nbrs_in_pid == max_neighbor_count:
                cells_for_nbr.append(cid)
        
        if cells_for_nbr:
            cell_dict = {}
            for cid in cells_for_nbr:
                max_nbr_cell = self.cell_manager.cells_dict[cid]
                # change the pid of the cell that was donated.
                max_nbr_cell.pid = pid
                cell_dict[cid] = max_nbr_cell
            # if all cells are being sent away, keep the last cid with self
            if len(cells) == len(cell_dict):
                del cell_dict[cid]
            particles = self.cell_manager.create_new_particle_copies(cell_dict)
            # update the neighbor information locally
            self.cell_manager.update_neighbor_information_local()
        else:
            logger.debug('No cells found for %d'%(pid))
            particles = {}

        return particles
    
    def _zero_request_particles(self):
        """
        Requests particles from processors with some particles.
        """
        arrays = self.cell_manager.arrays_to_bin
        comm = self.comm
        i = self.num_procs - 1
        req = {}
        done = False
        while i > 0 and done == False:
            pid = self.procs_to_communicate[i]
            np = self.particles_per_proc[pid]

            if np == 0:
                done = True
                continue

            num_particles = sum(map(ParticleArray.get_number_of_particles, arrays))
            req['num_particles'] = num_particles

            if num_particles > 0:
                req['need_particles'] = False
            else:
                req['need_particles'] = True

            comm.send(req, dest=pid, tag=TAG_LB_PARTICLE_REQUEST)
            data = comm.recv(source=pid, tag=TAG_LB_PARTICLE_REPLY)
            
            # add the particles in the parray
            particles = data['particles']
            self.cell_manager.add_local_particles_to_parray(particles)
            
            i -= 1

    def _zero_donate_particles(self):
        """
        Respond to a request for particles from a zero particle process.
        """
        comm = self.comm
        i = 0
        reply = {}
        done = False
        while i < self.num_procs and done == False:
            pid = self.procs_to_communicate[i]
            np = self.particles_per_proc[pid]

            if np > 0:
                done = True
                continue
            
            # receive the request from pid
            req = comm.recv(source=pid, tag=TAG_LB_PARTICLE_REQUEST)
            reply = self._process_zero_proc_request(pid, req)
            comm.send(reply, dest=pid, tag=TAG_LB_PARTICLE_REPLY)

            i += 1
    
    def _process_zero_proc_request(self, pid, request):
        """
        Constructs a reply for a request from a process with zero particles.
        """
        if request['need_particles'] == False:
            return {'particles':{}}
        
        num_particles_with_pid = request['num_particles']
        
        if num_particles_with_pid > 0:
            logger.warn('Invalid request from %d'%(pid))
            return {'particles':{}}

        particles = self._get_boundary_cells_to_donate(pid)

        return {'particles':particles}

    def _get_boundary_cells_to_donate(self, pid):
        """
        Gets one or more boundary cells to be donated to a process with zero
        particles. 

        **Algorithm**

            - find all boundary cells.
            - choose the one with the least number of neighbors as the cell to donate.

        """
        min_neighbor_id = None
        min_neighbors = 27

        for cid, c in self.cell_manager.cells_dict.iteritems():
            if c.pid != self.pid:
                continue
            pinfo = c.parallel_cell_info
            if pinfo.is_boundary_cell():
                logger.debug('Checking boundary cell : %s'%(cid))
                num_neighbors = len(pinfo.neighbor_cell_pids)
                if num_neighbors < min_neighbors:
                    min_neighbors = num_neighbors
                    min_neighbor_id = cid

        # we have the cell with the least number of neighbors.
        # extract the particles for this cell and return.
        if min_neighbor_id is None:
            # meaning there are no boundary cells to donate.
            # choose a cell with min number of local neighbors.
            min_neighbors = 27
            for cid, c in self.cell_manager.cells_dict.iteritems():
                if c.pid != self.pid:
                    continue
                pinfo = c.parallel_cell_info
                logger.debug('Cell %s with %s local nbrs'%(
                        cid, pinfo.num_local_neighbors))
                for nbrid in pinfo.neighbor_cell_pids:
                    logger.debug(nbrid)
                if pinfo.num_local_neighbors < min_neighbors:
                    min_neighbors = pinfo.num_local_neighbors
                    min_neighbor_id = cid

        min_nbr_cell = self.cell_manager.cells_dict.get(min_neighbor_id)

        if min_nbr_cell is not None:
            logger.debug('Transfering %s from proc %d to proc %d'%(min_neighbor_id,
                                                                   self.pid,
                                                                   pid))
            particles = self.cell_manager.create_new_particle_copies(
                {min_neighbor_id:min_nbr_cell})

            # mark that cell as a remote cell
            min_nbr_cell.pid = pid
            # remove the particles that were just extracted.
            #self.cell_manager.remove_remote_particles()
            
            # recompute the neighbor counts of all local cells.
            self.cell_manager.update_neighbor_information_local()
        else:
            logger.debug('No suitable cell found to donate to processor %d'%(
                    pid))
            particles = {}
        
        return particles
