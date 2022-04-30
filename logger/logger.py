from multiprocessing import Queue
from threading import Thread
import pickle
import time
import os
import asyncio
import sys

def add_dict(d,label,val):
    try:
        d[label].append(val)
    except KeyError:
        d[label] = [val]

def write_file(file_path,it,data):
    path = file_path+'/log'+".txt"
    if not(it is None):
        path = file_path+'/log-'+str(it)+".txt"
    with open(path,'wb') as f:
        pickle.dump(data,f)

def process_loop(file_path,queue,dtlog):
    data = {}
    it = 0
    write_file(file_path,it,data)
    tref = time.perf_counter()
    while True:
        m = queue.get()
        if m[0] == 'stop':
            write_file(file_path,it,data) #Because the plotter only looks at the -2 file written
            break
        elif m[0] == 'add':
            add_dict(data,m[1],m[2])
        elif m[0] == 'drop':
            del data[m[1]]
        if (time.perf_counter()-tref)**2>dtlog or m[0] == 'flush': #Saves regularly if it's updated
            write_file(file_path,it,data)
            tref = time.perf_counter()
            it += 1
    sys.exit()        

class Logger:
    def __init__(self,file_path,dtlog):
        self.file_path = file_path
        self.dtlog = dtlog

    def init(self):
        self.queue = Queue()
        self.process = Thread(target=process_loop,args=(self.file_path,self.queue,self.dtlog))
        self.process.daemon = True
    
    def start(self):
        self.process.start()
        print('started')

    def stop(self):
        command = ('stop',)
        self.queue.put(command)

    def add(self,name,val):
        command = ('add',name,val)
        self.queue.put(command)

    def drop(self,name):
        command = ('drop',name)
        self.queue.put(command)

    def flush(self):
        command = ('flush',)
        self.queue.put(command)
