import multiprocessing as mp
import time
import matplotlib.pyplot as plt
from matplotlib import animation
import pickle
import os
import numpy as np

def load_last_file(file_path):
    ldir = os.listdir(file_path)
    sorted_ldir = sorted(ldir,key=lambda x : (len(x),x))
    print(sorted_ldir)
    file_path_it = os.path.join(file_path,sorted_ldir[-1])
    with open(file_path_it,'rb') as f:
        return pickle.load(f)

def process_loop(actualization_rate,targets,file_path,pipe):
    fig,ax = plt.subplots()#plt.figure()
    #ax = plt.axes()
    line, = ax.plot([],[])
    def init():
        line.set_data([],[])
        return line,
    def animate(i):
        data = load_last_file(file_path)
        try:
            data = np.array(data[targets[0]])
            plt.xlim(0,len(data)*1.2)
            ymax = max(abs(data))*1.2
            plt.ylim(-ymax,ymax)
            print(len(data))
            line.set_data(np.arange(len(data)),data)
        except KeyError:
            pass
        return line,
    anim = animation.FuncAnimation(fig,animate,init_func=init,frames=None,interval=actualization_rate*1000,blit=True)
    plt.show()

class Plotter:
    def __init__(self,file_path,targets,actualization_rate=1):
        self.file_path = file_path
        self.actualization_rate = actualization_rate 
        self.targets = targets

    def start(self):
        self.pipe,pipe = mp.Pipe(duplex=True)
        self.process = mp.Process(target=process_loop,args=(self.actualization_rate,self.targets,self.file_path,pipe))
        self.process.start()

    def stop(self):
        self.process.close()