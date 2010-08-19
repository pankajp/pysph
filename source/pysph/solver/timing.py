"""
Simple module to find code execution times.
"""

import time

import logging
logger = logging.getLogger()

from pysph.base.carray import DoubleArray

###############################################################################
# `Timer` class.
###############################################################################
class Timer:
    """
    Class to measure and write code execution times.
    """
    def __init__(self, output_file_name='', write_after=100):
        """
        Constructor.
        """
        self.start_time = 0
        self.end_time = 0
        self.recorded_times = 0
        self.last_written_time = 0
        self.write_after = write_after
        self.elapsed_times = DoubleArray()
        self.output_file_name = output_file_name

    def start(self):
        """
        Set the start time of the timer.
        """
        self.start_time = time.time()

    def finish(self):
        """
        Set the end time of the timer.
        """
        self.end_time = time.time()
        last_measured_time = self.end_time - self.start_time
        self.elapsed_times.append(last_measured_time)
        self.recorded_times += 1
        self.dump()

    def dump(self):
        """
        Dump the data into the file.
        """
        if self.recorded_times%self.write_after != 0:
            return
        
        if self.output_file_name == '':
            msg = 'output_file_name not set'
            logger.error(msg)
            raise ValueError, msg

        f = open(self.output_file_name, 'a')
        
        for i in range(self.elapsed_times.length):
            f.write(str(i+self.last_written_time) + ' ' +
                    str(self.elapsed_times.get(i))+'\n')

        f.close()

        self.last_written_time = self.recorded_times
        self.elapsed_times.reset()

    def reset(self):
        """
        Reset counters etc.
        """
        self.recorded_times = 0
        self.last_written_time = 0
        self.start_time = 0
        self.end_time = 0
        self.elapsed_times.reset()
