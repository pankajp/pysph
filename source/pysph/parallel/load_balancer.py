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
from pysph.base.cell import CellManager, py_construct_immediate_neighbor_list

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
        self.pid = 0
        self.num_procs = 1
        self.particles_per_proc = []
        self.ideal_load = 0.
        self.threshold_ratio = 25.
        self.threshold_margin = 0.
        self.lb_max_iterations = 10
        self.upper_threshold = 0.
        self.lower_threshold = 0.
        self.load_difference = []
        self.prev_particle_count = []
        self.method = None
        
    def setup(self):
        """
        Sets up some internal data.
        """
        if self.setup_done == True:
            return

        self.parallel_controller = self.cell_manager.parallel_controller
        self.pid = self.parallel_controller.rank
        self.num_procs = self.parallel_controller.num_procs
        self.comm = self.parallel_controller.comm

        self.setup_done = True

    def load_balance(self, method=None, **args):
        """
        Calls the load_balance_func depending on skip_iteration if set
        """
        self.setup()
        if method is None:
            method = self.method
        
        if self.solver is None or (self.solver.current_iteration %
                                    self.skip_iteration == 0):
            if method is None or method == '':
                self.load_balance_func(**args)
            else:
                func = getattr(self, 'load_balance_func_'+method)
                func(**args)

    def load_balance_func(self):
        return self.load_balance_func_normal()
    
    def load_balance_func_normal(self):
        """
        Perform the load balancing.
        
        **Algorithm**
        
            - while load not balanced or lb iterations not exceeded.

                - Compute some statistics
                - Find the number of real particles in all processors.
                - Find the total number of particles.
                - Find the mean number of particles with each processor.
                - If number of particles with each processor is within a
                  particular threshold from the mean, load is balanced, exit.
                
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
            self.calc_load_thresholds(self.particles_per_proc)

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
            self.cell_manager.rebin_particles()
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
            self.cell_manager.proc_map.nbr_procs)
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
                cell_dict[cid] = [max_nbr_cell]
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
            logger.debug('Transferring %s from proc %d to proc %d'%(min_neighbor_id,
                                                                   self.pid,
                                                                   pid))
            particles = self.cell_manager.create_new_particle_copies(
                {min_neighbor_id:[min_nbr_cell]})

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
            #raise 'Not donating to zero proc'
        
        return particles

    def calc_load_thresholds(self, particles_per_proc):
        self.total_particles = sum(self.particles_per_proc)
        self.ideal_load = float(self.total_particles) / self.num_procs
        
        self.threshold_margin = self.ideal_load * self.threshold_ratio / 100.
        self.lower_threshold = self.ideal_load - self.threshold_margin
        self.upper_threshold = self.ideal_load + self.threshold_margin
        
        for i in range(self.num_procs):
            self.load_difference[i] = (self.particles_per_proc[i] - 
                                       self.ideal_load)

    def load_balance_func_serial(self, distr_func='single', **args):
        """
        Perform the load balancing serially by gathering all data on root proc
        
        **Algorithm**
        
            - on root proc

                - Compute some statistics
                - Find the number of real particles in all processors.
                - Find the total number of particles.
                - Find the mean number of particles with each processor.
                - If number of particles with each processor is within a
                  particular threshold from the mean, load is balanced, exit.
                
                - Sort processor ids in increasing order of number of particles
                  with them. In case of multiple processors having the same
                  number of particles, arrange them in ascending order of pid.
                
                - If there are some processors with 0 particles, communication
                  among all processors.
                - If no such processors are there, each processor shall
                  communicate with adjacent neighbors.
                
                - collect all cells and number of particles on each proc on root
                - distribute particles on root proc using same algorithm as for
                    distributed load balancing
            - send the info to send/recv cells to all procs
                
            - BARRIER.
            - bin particles top down.
            - update processor map.
            - update neighbor information.

            - lb_iterations += 1
        """
        redistr_func = getattr(self, 'load_redistr_'+distr_func)
        self.balancing_done = False
        current_balance_iteration = 0
        self.load_difference = [0] * self.num_procs
        
        self._gather_cell_particles_info()
        
        old_distr = {}
        for proc_no, cells in enumerate(self.proc_cell_np):
            for cellid in cells:
                old_distr[cellid] = proc_no
        self.old_distr = old_distr
        self.cell_proc = {}
        self.cell_proc.update(old_distr)
        #print '(%d)'%self.pid, self.cell_proc
        self.cell_nbr_proc = self.construct_nbr_cell_info(self.cell_proc)
        
        while self.balancing_done == False and self.pid == 0:
            logger.info('Load Balance iteration %d -------------------' % (
                    current_balance_iteration))

            if current_balance_iteration >= self.lb_max_iterations:
                self.balancing_done = True
                logger.info('MAX LB ITERATIONS EXCEEDED')
                continue
                
            self.load_balance_serial_iter(redistr_func, **args)
            current_balance_iteration += 1
        
        # do the actual transfer of particles now
        self.redistr_cells(self.old_distr, self.cell_proc)
        logger.info('load distribution : %s : '%(str(set(self.cell_proc.values()))) + str(self.particles_per_proc))
        # update the cell information.
        self.cell_manager.remove_remote_particles()
        self.cell_manager.rebin_particles()
        self.cell_manager.glb_update_proc_map()
        self.cell_manager.update_cell_neighbor_information()
        
        logger.info('waiting for lb to finish')
        self.comm.Barrier()
        
        if logger.getEffectiveLevel() <= 20: # only for level <= INFO
            cell_np = {}
            np = 0
            for cellid, cell in self.cell_manager.cells_dict.items():
                cell_np[cellid] = cell.get_number_of_particles()
                np += cell_np[cellid]
            
            logger.info('(%d) %d particles in %d cells' % (self.pid, np, len(cell_np)))
    
    def load_balance_serial_iter(self, redistr_func, **args):
        # get the number of particles with each process.
        #self.particles_per_proc = self.collect_num_particles()
        self.calc_load_thresholds(self.particles_per_proc)
        min_diff = min(self.load_difference)
        max_diff = max(self.load_difference)
        
        if (abs(min_diff) < self.threshold_margin and max_diff < 
            self.threshold_margin):
            self.balancing_done = True
            logger.info('BALANCE ACHIEVED')
            logger.debug('Num particles are : %s' % (self.particles_per_proc))
            return

        if self.particles_per_proc == self.prev_particle_count and self.pid == 0:
            # meaning that the previous load balancing iteration did not
            # change the particle counts, we do not do anything now.
            self.balancing_done = True
            logger.info('Load unchanged')
            return
        
        if logger.getEffectiveLevel() <= 20: # only for level <= INFO
            logger.debug('Total particles : %d' % (self.total_particles))
            logger.debug('Ideal load : %d' % (self.ideal_load))
            logger.debug('Load Difference : %s' % (self.load_difference))
            logger.info('Particle counts : %s' % (self.particles_per_proc))
            logger.debug('Threshold margin: %f' % (self.threshold_margin))
            logger.debug('Upper threshold : %f' % (self.upper_threshold))
            logger.debug('Lower threshold : %f' % (self.lower_threshold))
        
        if not self.balancing_done:
            # store the old particle counts in prev_particle_count
            self.prev_particle_count[:] = self.particles_per_proc
            self.cell_proc, self.particles_per_proc = redistr_func(
                                    self.cell_proc, self.proc_cell_np, **args)
    
    def _gather_cell_particles_info(self):
        self.particles_per_proc = [0] * self.num_procs
        cell_np = {}
        for cellid, cell in self.cell_manager.cells_dict.items():
            cell_np[cellid] = cell.get_number_of_particles()
        self.proc_cell_np = self.comm.gather(cell_np, root=0)
        #print '(%d)'%self.pid, self.proc_cell_np
        
        if self.proc_cell_np is None:
            self.proc_cell_np = []        
        
        for i, c in enumerate(self.proc_cell_np):
            for cnp in c.values():
                self.particles_per_proc[i] += cnp
        logger.debug('(%d) %s' % (self.pid, str(self.particles_per_proc)))
        
        self.particles_per_proc = self.comm.bcast(self.particles_per_proc, root=0)
        logger.debug('(%d) %s' % (self.pid, str(self.particles_per_proc)))
    
    def redistr_cells(self, old_distr, new_distr):
        """ redistribute cells in the procs as per the new distr,
        using old_distr to determine the incremental data to be communicated
        old_distr and new_distr are used only on the root proc
        old_distr and new_distr are dict of cid:pid and need only contain
        changed cells
        """
        logging.debug('redistributing cells')
        
        sends = [[[] * self.num_procs for i in range(self.num_procs)] for j in range(self.num_procs)]
        recvs = [[[] * self.num_procs for i in range(self.num_procs)] for j in range(self.num_procs)]
        for cid, opid in old_distr.iteritems():
            npid = new_distr[cid]
            if opid != npid:
                recvs[npid][opid].append(cid)
                sends[opid][npid].append(cid)
        
        sends = self.comm.scatter(sends, root=0)
        recvs = self.comm.scatter(recvs, root=0)
        
        # now each proc has all the cells it needs to send/recv from other procs
        
        logging.debug('sends' + str([len(i) for i in sends]))
        logging.debug('recvs' + str([len(i) for i in recvs]))
        
        # greater pid will recv first
        for i in range(self.pid):
            self.recv_particles(recvs[i], i)
            self.send_particles(sends[i], i)
        
        # smaller pid will send first
        for i in range(self.pid + 1, self.num_procs):
            self.send_particles(sends[i], i)
            self.recv_particles(recvs[i], i)
        
        logging.debug('redistribution of cells done')
    
    def load_redistr_single(self, cell_proc=None, proc_cell_np=None, **args):
        """ The load balance algorithm running on root proc
        
        The algorithm is same as the parallel normal load balancing algorithm,
        except zero proc handling that is run completely on the root proc
        """
        self.procs_to_communicate = self._get_procs_to_communicate(
            self.particles_per_proc, range(self.num_procs))
        self.procs_to_communicate = numpy.argsort(self.particles_per_proc)[::-1]
        num_procs = len(self.procs_to_communicate)
        
        # pass1 pid = pid, pass2 pid = pidr
        for i in range(num_procs):
            pid = self.procs_to_communicate[-i - 1]
            for j in range(num_procs - i - 1):
                pidr = self.procs_to_communicate[j]
                self.single_lb_transfer_cells(pid, pidr)
        logger.debug('load_redistr_single done')
        return self.cell_proc, self.particles_per_proc
    
    def load_redistr_auto(self, cell_proc=None, proc_cell_np=None, **args):
        """ load redistribution by automatic selection of method
        
        If only one proc has all the particles, then use the
        load_redistr_geometric method, else use load_redistr_simple
        """
        non_zeros = len([1 for p in self.particles_per_proc if p > 0])
        if non_zeros == 1:
            logger.info('load_redistr_auto: geometric')
            cell_proc, np_per_proc = self.load_redistr_geometric(self.cell_proc, self.proc_cell_np)
            self.balancing_done = False
            self.cell_nbr_proc = self.construct_nbr_cell_info(cell_proc)
            cell_np = {}
            for proc,c_np in enumerate(self.proc_cell_np):
                cell_np.update(c_np)
            self.proc_cell_np = [{} for i in range(self.num_procs)]
            for cid,pid in cell_proc.iteritems():
                self.proc_cell_np[pid][cid] = cell_np[cid]
            return cell_proc, np_per_proc
        else:
            logger.info('load_redistr_auto: serial')
            return self.load_redistr_single(self.cell_proc, self.proc_cell_np)
    
    def single_lb_transfer_cells(self, pid, pidr):
        """ Allocate particles from proc pidr to proc pid (on root cell) """
        num_particles = self.particles_per_proc[pid]

        if num_particles < self.ideal_load:
            need_particles = True
        else:
            need_particles = True

        num_particlesr = self.particles_per_proc[pidr]
        
        if num_particles == 0 and num_particlesr > 1:
            # send a cell to zero proc
            cells = self._get_cells_for_zero_proc(pid, self.proc_cell_np[pidr])
            for cid in cells:
                self._update_cell_pid_info(cid, pidr, pid)
            
            return cells
        
        logger.debug('%d %d %d %d transfer' % (pid, num_particles, pidr, num_particlesr))

        if need_particles == False:
            logger.debug('%d request for NO particles' % (pid))
            return []

        # check if pid has more particles than pidr
        if num_particles >= num_particlesr:
            logger.debug('%d has more particles that %d' % (pid, self.pid))
            return []
        
        # if number of particles in pidr is within the threshold, do not donate
        # particles
        if abs(self.ideal_load - num_particlesr) < self.threshold_margin:
            if (not (num_particlesr - num_particles) > self.threshold_margin):
                logger.debug('Need not donate - not overloaded')
                return []

        # if pidr has only one cell, do not donate
        if len(self.proc_cell_np[pidr]) == 1:
            logger.debug('Have only one cell - will not donate')
            return []

        # get one or more cells to send to pid
        cells = self._get_cells_for_neighbor_proc(pid, self.proc_cell_np[pidr])
        for cid in cells:
            self._update_cell_pid_info(cid, pidr, pid)
        
        return cells
    
    def recv_particles(self, cells, pid):
        """ recv particles from proc pid """
        # do not communicate if nothing is to be transferred
        if not cells: # cells is empty
            return
        logger.debug('Receiving particles in %d cells from %d' % (len(cells), pid))

        particles = self.comm.recv(source=pid, tag=TAG_LB_PARTICLE_REPLY)
        self.cell_manager.add_local_particles_to_parray(particles)
        
        logger.debug('Received particles from %d' % (pid))
    
    def send_particles(self, cells, pid):
        """ send particles in cells to proc pid """
        # do not communicate if nothing is to be transferred
        if not cells: # cells is empty
            return
        logger.debug('Sending particles in %d cells to %d' % (len(cells), pid))

        particles = self._build_particle_to_send_from_cells(cells, pid)
        self.comm.send(particles, dest=pid, tag=TAG_LB_PARTICLE_REPLY)

        logger.debug('Sent particles to %d' % (pid))
    
    def _build_particle_to_send_from_cells(self, cells, pid):
        """
        Build the reply to be sent in response to a request.
        Returns particles cells to be moved to pid for processing
        """
        cell_dict = {}
        for cid in cells:
            cell = self.cell_manager.cells_dict[cid]
            # change the pid of the cell that was donated.
            cell.pid = pid
            cell_dict[cid] = [cell]
        particles = self.cell_manager.create_new_particle_copies(cell_dict)
        # update the neighbor information locally
        self.cell_manager.update_neighbor_information_local()
        
        return particles
    
    def _get_cells_for_zero_proc(self, pid, cells):
        """ return a cell to be sent to nbr zero proc `pid`
        cells is the sequence of cells from which to choose the cells to send
        """
        max_empty_count = -1
        min_nbrs = 27
        cells_for_nbr = []

        for cid in cells:
            empty_count = self.cell_nbr_proc[cid].get(-1, 0)
            
            if empty_count > max_empty_count:
                max_empty_count = empty_count
                min_nbrs = len(self.cell_nbr_proc[cid])-1
                cells_for_nbr = [cid]
            elif empty_count == max_empty_count:
                nbrs = len(self.cell_nbr_proc[cid])-1
                if nbrs < min_nbrs:
                    max_empty_count = empty_count
                    min_nbrs = nbrs
                    cells_for_nbr = [cid]
        
        return cells_for_nbr
    
    def _get_cells_for_neighbor_proc(self, pid, cells):
        """ return cells to be sent to nbr proc `pid`
        cells is the sequence of cells from which to choose the cells to send
        """
        max_neighbor_count = -1
        cells_for_nbr = []

        for cid in cells:
            cpid = self.cell_proc[cid]
            local_nbr_count = self.cell_nbr_proc[cid].get(cpid, 0)
            remote_nbr_count = 26 - self.cell_nbr_proc[cid].get(-1, 0) - local_nbr_count
            
            if remote_nbr_count == 0:
                #logger.debug('%s has no remote nbrs'%(cid))
                continue

            num_nbrs_in_pid = self.cell_nbr_proc[cid].get(pid)
            
            if not num_nbrs_in_pid:
                continue

            if num_nbrs_in_pid > max_neighbor_count:
                max_neighbor_count = num_nbrs_in_pid
                cells_for_nbr = [cid]
            elif num_nbrs_in_pid == max_neighbor_count:
                cells_for_nbr.append(cid)
        
        if not cells_for_nbr:
            logger.debug('No cells found for %d' % (pid))

        return cells_for_nbr
    
    @classmethod
    def construct_nbr_cell_info(self, cell_proc):
        """ Construct and return the dict of cid:{pid:nnbr} having the neighbor
        pid information for each cell """
        cell_nbr_proc = {} # cid:{pid:nnbr}
        for cid in cell_proc:
            nbrs = []
            nbrcnt = {}
            py_construct_immediate_neighbor_list(cid, nbrs, False)
            for nbr in nbrs:
                p = cell_proc.get(nbr, -1) # -1 is count of missing neighbors
                nbrcnt[p] = nbrcnt.get(p, 0) + 1
            cell_nbr_proc[cid] = nbrcnt
        return cell_nbr_proc
    
    def _update_cell_pid_info(self, cid, old_pid, new_pid):
        """ Update the cell_nbr_proc dict to reflect a change in the pid of
        cell cid to new_pid """
        
        #print cid, old_pid, new_pid
        self.cell_proc[cid] = new_pid
        
        cell_nbr_proc = self.cell_nbr_proc
        nbrs = []
        py_construct_immediate_neighbor_list(cid, nbrs, False)
        for nbr in nbrs:
            nbr_info = cell_nbr_proc.get(nbr) 
            if nbr_info is not None:
                nbr_info[old_pid] -= 1
                nbr_info[new_pid] = nbr_info.get(new_pid, 0) + 1
        
        self.proc_cell_np[new_pid][cid] = self.proc_cell_np[old_pid][cid]
        del self.proc_cell_np[old_pid][cid]
        
        cell_np = self.proc_cell_np[new_pid][cid]
        self.particles_per_proc[old_pid] -= cell_np
        self.particles_per_proc[new_pid] += cell_np
    
    
    ###########################################################################
    # simple method to assign some cells to all procs based on geometry
    # subdivision. The distribution is unsuitable as load balancer,
    # but may provide a good method to initiate laod balancing
    ###########################################################################
    
    def load_redistr_geometric(self, cell_proc, proc_cell_np, allow_zero=False, **args):
        """ distribute cell_np to processors in a simple geometric way
        
        **algorithm**
         # get the distribution size of each dimension using `get_distr_size()`
             based on the domain size of the cell_np     # divide the domain into rectangular grids
         # assign cell_npto each processor
         # check empty processors and divide cell_npin processor having more than
           average cell_npto the empty processors
        """
        num_procs = len(proc_cell_np)
        cell_np = {}
        for cnp in proc_cell_np:
            cell_np.update(cnp)
        proc_cells, proc_num_particles = self.distribute_particles_geometric(
                                                cell_np, num_procs, allow_zero)
        self.balancing_done = True
        return self.get_cell_proc(proc_cells=proc_cells), proc_num_particles

    @staticmethod
    def get_distr_sizes(l=1., b=1., h=1., num_procs=12):
        """return the number of clusters to be generated along each dimension
        
        l,b,h are the size of the domain
        return: s = ndarray of size 3 = number of divisions along each dimension
        s[0]*s[1]*s[2] >= num_procs"""
        x = numpy.array([l,b,h], dtype='float')
        compprod = numpy.cumprod(x)[-1]
        fac = (float(num_procs)/compprod)**(1.0/3)
        s = x*fac
        s = numpy.ceil(s)
        cont = True
        while cont:
            ss = numpy.argsort(s)
            if (s[ss[2]]-1)*(s[ss[1]])*(s[ss[0]]) >= num_procs:
                s[ss[2]] -= 1
                continue
            elif (s[ss[2]])*(s[ss[1]]-1)*(s[ss[0]]) >= num_procs:
                s[ss[1]] -= 1
                continue
            elif (s[ss[2]])*(s[ss[1]])*(s[ss[0]]-1) >= num_procs:
                s[ss[0]] -= 1
                continue
            else:
                cont = False
        #print 'sizes: %s'%(str(s))
        #print distortion(*s/x)
        return s
    
    @staticmethod
    def distribute_particles_geometric(cell_np, num_procs, allow_zero=False):
        """ distribute cell_npto processors in a simple way
        
        **algorithm**
         # get the distribution size of each dimension using `get_distr_size()`
             based on the domain size of the cell_np
         # divide the domain into rectangular grids
         # assign cell_npto each processor
         # check empty processors and divide cell_npin processor having more than
           average cell_np to the empty processors
        """
        num_cells = len(cell_np)
        cell_arr = numpy.empty((num_cells, 3))
        num_particles_arr = numpy.empty((num_cells,), dtype='int')
        for i,cell_id in enumerate(cell_np):
            cell_arr[i,0] = cell_id.x
            cell_arr[i,1] = cell_id.y
            cell_arr[i,2] = cell_id.z
            num_particles_arr[i] = cell_np[cell_id]
        
        np_per_proc = sum(num_particles_arr)/num_procs
        
        lmin = numpy.min(cell_arr[:,0])
        bmin = numpy.min(cell_arr[:,1])
        hmin = numpy.min(cell_arr[:,2])
        # range of cells in each dimension
        l = numpy.max(cell_arr[:,0])+1 - lmin
        b = numpy.max(cell_arr[:,1])+1 - bmin
        h = numpy.max(cell_arr[:,2])+1 - hmin
        
        # distribution sizes in each dimension
        s = LoadBalancer.get_distr_sizes(l,b,h,num_procs)
        
        ld = l/s[0]
        bd = b/s[1]
        hd = h/s[2]
        
        # allocate regions to procs
        
        # deficit of actual processes to allocate
        deficit = int(numpy.cumprod(s)[-1] - num_procs)
        # sorted s
        ss = numpy.argsort(s)
        # reversed dict (value to index)
        rss = numpy.empty(len(ss), dtype='int')
        for i,si in enumerate(ss):
            rss[si] = i
        proc = 0
        proc_cells = [[] for i in range(num_procs)]
        proc_map = {}
        done = False
        for i in range(int(s[ss[0]])):
            for j in range(int(s[ss[1]])):
                for k in range(int(s[ss[2]])):
                    if done:
                        done = False
                        continue
                    proc_map[tuple(numpy.array((i,j,k),dtype='int')[rss])] = proc
                    proc += 1
                    if deficit > 0 and k==0:
                        deficit -= 1
                        proc_map[tuple(numpy.array((i,j,k+1),dtype='int')[rss])] = proc-1
                        done = True
        
        # allocate cell_np to procs
        proc_num_particles = [0 for i in range(num_procs)]
        for i,cell_id in enumerate(cell_np):
            index = (int((cell_id.x-lmin)//ld), int((cell_id.y-bmin)//bd), int((cell_id.z-hmin)//hd))
            proc_cells[proc_map[index]].append(cell_id)
            proc_num_particles[proc_map[index]] += cell_np[cell_id]

        # return the distribution if procs with zero cells are permitted
        if allow_zero:
            return proc_cells, proc_num_particles
        
        # add cell_np to empty procs
        proc_particles_s = numpy.argsort(proc_num_particles)
        empty_procs = [proc for proc,np in enumerate(proc_num_particles) if np==0]
        i = num_procs - 1
        while len(empty_procs) > 0:
            nparts = int(min(numpy.ceil(
                            proc_num_particles[proc_particles_s[i]]/float(np_per_proc)),
                         len(empty_procs)))
            cells = proc_cells[proc_particles_s[i]]
            ncells = int((len(cells)/float(nparts+1)))
            
            proc_cells[proc_particles_s[i]] = []
            cells_sorted = sorted(cells, key=hash)
            
            for j in range(nparts):
                cells2send = cells_sorted[j*ncells:(j+1)*ncells]
                proc_cells[empty_procs[j]][:] = cells2send
                for cid in cells2send:
                    proc_num_particles[empty_procs[j]] += cell_np[cid]
                    proc_num_particles[proc_particles_s[i]] -= cell_np[cid]                    
            proc_cells[proc_particles_s[i]][:] = cells_sorted[(j+1)*ncells:]
            empty_procs[:nparts] = []
            i -= 1
        return proc_cells, proc_num_particles
    ###########################################################################
    
    @classmethod
    def get_cell_proc(self, proc_cells):
        cell_proc = {}
        for proc, cids in enumerate(proc_cells):
            cell_proc.update(dict.fromkeys(cids, proc))
        return cell_proc
    
    @classmethod
    def get_load_imbalance(self, particles_per_proc):
        """return the imbalance in the load distribution = (max-avg)/max"""
        total = sum(particles_per_proc)
        avg = float(total)/len(particles_per_proc)
        mx = max(particles_per_proc)
        return (mx-avg)/mx
    
    @classmethod
    def get_quality(self, cell_nbr_proc, cell_proc, num_procs, ndim):
        num_cells = len(cell_nbr_proc)
        cells_nbr = cells_nbr_proc = procs_nbr = 0
        max_nbrs = (3**ndim-1)
        proc_nbrs = [set() for i in range(num_procs)]
        for cid,proc_np in cell_nbr_proc.iteritems():
            pid = cell_proc[cid]
            cells_nbr += 26 - proc_np.get(-1, 0) - proc_np.get(pid, 0)
            cells_nbr_proc += len(proc_np) - 1 - (-1 in proc_np)
            proc_nbrs[pid].update(proc_np)
        for pid, proc_nbrs_data in enumerate(proc_nbrs):
            proc_nbrs_data.remove(-1)
            proc_nbrs_data.remove(pid)
        #print proc_nbrs
        fac = num_cells**((ndim-1.0)/ndim) * max_nbrs
        cells_nbr = cells_nbr / fac
        cells_nbr_proc = cells_nbr_proc / fac
        procs_nbr = sum([len(i) for i in proc_nbrs])/float(num_procs)
        return cells_nbr, cells_nbr_proc, procs_nbr
    
    @classmethod
    def get_metric(self, cell_proc, particles_per_proc, ndim=None):
        """ return a performance metric for the current load distribution """
        if ndim is None:
            # FIXME: detect the dimension of the problem
            ndim = 2
        imbalance = self.get_load_imbalance(particles_per_proc)
        num_procs = len(particles_per_proc)
        quality = self.get_quality(self.construct_nbr_cell_info(cell_proc), cell_proc, num_procs, ndim)
        return (imbalance,) + quality
    
    @classmethod
    def plot(self, proc_cells, show=True, save_filename=None):
        try:
            from enthought.mayavi import mlab
        except:
            logger.critical('LoadBalancer.plot(): need mayavi to plot')
            return
        cell_idx = {}
        #print [len(i) for i in proc_cells]
        i = 0
        for procno, proccells in enumerate(proc_cells):
            for cell_id in proccells:
                cell_idx[cell_id] = i
                i += 1
        num_cells = i
        
        x = [0] * num_cells
        y = [0] * num_cells
        z = [0] * num_cells
        p = [0] * num_cells
        i = 0
        for procno, proccells in enumerate(proc_cells):
            for cell_id in proccells:
                x[cell_idx[cell_id]] = cell_id.x
                y[cell_idx[cell_id]] = cell_id.y
                z[cell_idx[cell_id]] = cell_id.z
                p[cell_idx[cell_id]] = procno
                i += 1
        
        figure = mlab.figure(0, size=(1200,900))
        plot = mlab.points3d(x, y, z, p, mode='cube', colormap='jet',
                             scale_mode='none', scale_factor=0.8, figure=figure)
        engine = mlab.get_engine()
        scene = engine.scenes[0]
        scene.scene.parallel_projection = True
        #scene.scene.camera.view_up = [0.0, 1.0, 0.0]
        mlab.view(0,0)
        if save_filename:
            mlab.savefig(save_filename, figure=figure)
        if show:
            mlab.show()
    
    @classmethod
    def distribute_particle_arrays(cls, particle_arrays, num_procs, cell_size, max_iter=100, distr_func='single', **args):
        """Convenience function to distribute given particles into procs
        
        Uses the load_balance_func_serial() function of LoadBalancer class to
        distribute the particles. Balancing methods can be changed by passing
        the same `args` as to the load_balance_func_serial method
        """
        try:
            from load_balancer_mkmeans import LoadBalancerMKMeans as LoadBalancer
        except ImportError:
            try:
                from load_balancer_sfc import LoadBalancerSFC as LoadBalancer
            except ImportError:
                pass
        #print LoadBalancer
        lb = LoadBalancer()
        lb.pid = 0
        lb.num_procs = num_procs
        lb.lb_max_iteration = max_iter
        
        redistr_func = getattr(lb, 'load_redistr_'+distr_func)
        #print redistr_func
        lb.load_difference = [0] * lb.num_procs
        
        cm = CellManager(particle_arrays, cell_size, cell_size)
        #print 'num_cells=', len(cm.cells_dict), cm.cell_size
        
        lb.particles_per_proc = [0] * lb.num_procs
        cell_np = {}
        for cellid, cell in cm.cells_dict.items():
            cell_np[cellid] = cell.get_number_of_particles()
        lb.proc_cell_np = [{} for i in range(num_procs)]
        lb.proc_cell_np[0] = cell_np
        #print '(%d)'%self.pid, self.proc_cell_np
        
        for i, c in enumerate(lb.proc_cell_np):
            for cnp in c.values():
                lb.particles_per_proc[i] += cnp
        
        old_distr = {}
        for proc_no, cells in enumerate(lb.proc_cell_np):
            for cellid in cells:
                old_distr[cellid] = proc_no
        lb.old_distr = old_distr
        lb.cell_proc = {}
        lb.cell_proc.update(old_distr)
        #print '(%d)'%self.pid, self.cell_proc
        lb.cell_nbr_proc = lb.construct_nbr_cell_info(lb.cell_proc)
        
        lb.balancing_done = False
        current_balance_iteration = 0
        while lb.balancing_done == False and current_balance_iteration < max_iter:
            #print '\riteration', current_balance_iteration, 
            lb.load_balance_serial_iter(redistr_func, **args)
            current_balance_iteration += 1
        
        na = len(cm.arrays_to_bin)
        particle_arrays_per_proc = [[ParticleArray() for j in range(na)] for i in range(num_procs)]
        
        cells_dict = cm.cells_dict
        a2b = cm.arrays_to_bin
        for cid, proc in lb.cell_proc.iteritems():
            cell = cells_dict[cid]
            pid_list = []
            cell.get_particle_ids(pid_list)
            for i in range(na):
                arr = particle_arrays_per_proc[proc][i]
                arr.append_parray(a2b[i].extract_particles(pid_list[i]))

                arr.set_name(a2b[i].name)
                arr.set_particle_type(a2b[i].particle_type)
        
        return particle_arrays_per_proc
    
    @classmethod
    def distribute_particles(cls, particle_array, num_procs, cell_size, max_iter=100, distr_func='auto', **args):
        """Same as distribute_particle_arrays but for a single particle array
        """
        if isinstance(particle_array, (ParticleArray,)):
            is_particle_array = True
            pas = [particle_array]
        else:
            # assume particle_array is list of particle_arrays
            is_particle_array = False
            pas = particle_array
        ret =  cls.distribute_particle_arrays(pas, num_procs, cell_size, max_iter, distr_func, **args)
        if is_particle_array:
            ret = [i[0] for i in ret]
        return ret
    
