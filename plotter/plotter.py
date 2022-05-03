import multiprocessing as mp
import time
from jax import linear_transpose
import matplotlib.pyplot as plt
from matplotlib import animation
import pickle
import os
import numpy as np

def load_last_file(file_path):
    ldir = os.listdir(file_path)
    sorted_ldir = sorted(ldir,key=lambda x : (len(x),x))
    file_name = sorted_ldir[0]
    if len(sorted_ldir)>=2:
        file_name = sorted_ldir[-2]
    file_path_it = os.path.join(file_path,file_name)
    with open(file_path_it,'rb') as f:
        return pickle.load(f)

def process_loop(actualization_rate,plot_classes,file_path,pipe):
    #Build figure
    rows = int(len(plot_classes)**0.5)+1
    cols = int(len(plot_classes)**0.5)+1
    fig,axes = plt.subplots(rows,cols)#plt.figure()
    axes = np.reshape(axes,(-1))

    #Build plots
    lines = []
    plots = []
    for plot_class,ax in zip(plot_classes,axes):
        plot = plot_class(ax)
        plot.init_legend()
        plots.append(plot)
        lines += plot.lines

    def animate(i):
        data_dict = load_last_file(file_path)
        lines = []
        for plot in plots:
            plot.show(data_dict)
            lines += plot.flatten_lines
        return lines

    anim = animation.FuncAnimation(fig,animate,frames=None,interval=actualization_rate*1000,blit=True)
    plt.legend()
    plt.show()

def process_loop2(actualization_rate,plot_classes,file_path,pipe):
    rows = int(len(plot_classes)**0.5)+1
    cols = int(len(plot_classes)**0.5)+1
    fig,axes = plt.subplots(rows,cols)#plt.figure()
    axes = np.reshape(axes,(-1))
    lines = []
    flatten_lines = []
    for i,ax in enumerate(axes):
        lines.append([])
        plt.subplot(rows,cols,i+1)
        for j in ti:
            if j in ['action']:
                line, = ax.plot([],[],label=j,marker='.',lw=0.)
            else:
                line, = ax.plot([],[],label=j)
            lines[i].append(line)
            flatten_lines.append(line)
        plt.legend()

    minis = [0]*len(plot_classes)
    maxis = [1]*len(plot_classes)

    def init():
        for bl in lines:
            for l in bl:
                l.set_data([],[])
        return flatten_lines

    def animate(i):
        data_dict = load_last_file(file_path)
        for i,(ax,bl,bt) in enumerate(zip(axes,lines,targets)):
            for l,t in zip(bl,bt):
                try:
                    data = np.array(data_dict[t])
                    ax.set_xlim(0,len(data)*1.1)
                    minis[i] = min(minis[i],min(data))
                    maxis[i] = max(maxis[i],max(data))
                    r = maxis[i]-minis[i]
                    ax.set_ylim(minis[i]-r*0.1,maxis[i]+r*0.1)
                    l.set_data(np.arange(len(data)),data)
                except KeyError:
                    pass
        return flatten_lines

    anim = animation.FuncAnimation(fig,animate,init_func=init,frames=None,interval=actualization_rate*1000,blit=False)
    plt.show()

class Plotter:
    def __init__(self,file_path,plots,actualization_rate=1):
        self.file_path = file_path
        self.actualization_rate = actualization_rate
        self.plot_classes = plots

    def start(self):
        self.pipe,pipe = mp.Pipe(duplex=True)
        self.process = mp.Process(target=process_loop,args=(self.actualization_rate,self.plot_classes,self.file_path,pipe))
        self.process.start()

    def stop(self):
        self.process.terminate()
