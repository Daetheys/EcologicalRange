import matplotlib.pyplot as plt
import numpy as np

from plotter.lines.line import Line
import numpy

def scatter(*args,**kwargs):
    return lambda lbl : Scatter(*args,label=lbl,**kwargs)

class Scatter(Line):
    def __init__(self,*args,cmap='Viridis',**kwargs):
        super().__init__(*args,**kwargs)
        self.lines = {"main":self.ax.imshow(np.zeros((1,20)),animated=True)}
        self.cmap = cmap

    def set_data(self,x,y,update_ranges=True):
        y = y[...,-1]*8
        y = 1/np.exp(y[:,:,None] - y[:,None,:]).sum(axis=1)
        self.lines['main'] = self.ax.imshow(y.transpose(),vmin=0,vmax=1,cmap=self.cmap,aspect=y.shape[0]/y.shape[1]/2.2)
        if update_ranges:
            self.update_ranges(x,[0,y.shape[1]])
