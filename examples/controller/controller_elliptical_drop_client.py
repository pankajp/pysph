""" Example file showing the use of solver controller and various interfaces

Usage:
    Run this file after running the `controller_elliptical_drop.py` example file
    A matplotlib plot window will open showing the current position of all
    the particles and colored according to their velocities. The plot is updated
    every second. This is based on the multiprocessing interface
    A browser window is also opened which displays the various solver properties
    and also allows you to change then. It is based on the xml-rpc interface
"""

import matplotlib
matplotlib.use('GTKAgg') # do this before importing pylab
import matplotlib.pyplot as plt

import gobject # for the gobject timer

import time
import numpy
import webbrowser
import xmlrpclib

from pysph.solver.solver_interfaces import MultiprocessingClient


def test_interface(controller):
    print 't1', controller.get('dt')
    print 't2', controller.get_dt()
    
    task_id = controller.pause_on_next()
    print task_id
    time.sleep(1)
    print 'count', controller.get_count()
    time.sleep(1)
    
    # main thread is stopped; count should still be same
    print 'count2', controller.get_count()
    controller.cont()
    
    # main thread now still running; count should have increased
    time.sleep(1)
    print 'count3', controller.get_count()
    
    task_id = controller.get_particle_array_names()
    pa_names = controller.get_result(task_id) # blocking call
    print 'pa_names', task_id, pa_names
    
    print controller.get_status()
    

def test_XMLRPC_interface(address='http://localhost:8900/'):
    client = xmlrpclib.ServerProxy(address, allow_none=True)
    
    print client.system.listMethods()
    # client has all methods of `control` instance
    print client.get_t()
    print 'xmlrpcclient:count', client.get('count')
    
    test_interface(client)
    return client


def test_web_interface(address='http://127.0.0.1:8900/controller_elliptical_drop_client.html'):
    webbrowser.open(url=address)


def test_multiprocessing_interface(address=('localhost',8800), authkey='pysph'):
    client = MultiprocessingClient(address, authkey)
    
    controller = client.controller
    task_id = controller.get_particle_array_names()
    pa_names = controller.get_result(task_id) # blocking call
    task_id = controller.get_named_particle_array(pa_names[0])
    print task_id
    print controller.get_result(task_id)
    
    return controller


def test_plot(controller):
    task_id = controller.get_particle_array_names()
    pa_name = controller.get_result(task_id)[0] # blocking call
    pa = controller.get_result(controller.get_named_particle_array(pa_name))
    
    #plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line = ax.scatter(pa.x, pa.y, c=numpy.hypot(pa.u,pa.v))
    
    global t
    t = time.time()
    def update():
        global t
        t2 = time.time()
        dt = t2 - t
        t = t2
        print 'count:', controller.get_count(), '\ttimer time:', dt,
        pa = controller.get_result(controller.get_named_particle_array(pa_name))
    
        line.set_offsets(zip(pa.x, pa.y))
        line.set_array(numpy.hypot(pa.u,pa.v))
        fig.canvas.draw()
        
        print '\tresult & draw time:', time.time()-t
        
        return True
    
    update()
    
    # due to some gil issues in matplotlib, updates work only when
    # mouse is being hovered over the plot area (or a key being pressed)
    # when using python threading.Timer. Hence gobject.timeout_add
    # is being used instead
    gobject.timeout_add_seconds(1, update)
    plt.show()


def test_main():
    
    test_XMLRPC_interface()
    controller = test_multiprocessing_interface()
    
    test_web_interface()
    
    test_plot(controller)


if __name__ == '__main__':
    test_main()
