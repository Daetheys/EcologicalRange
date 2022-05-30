import matplotlib.pyplot as plt
import numpy as np

class Line:
    def __init__(self,ax,color=None,lw=1.,label=None):
        self.ax = ax
        
        self.color = color
        self.lw = lw
        self.label = label
        
        self.lines = {"main":self.ax.plot([],[],c=color,lw=lw,label=label)[0]}
        self.mini_y = None
        self.maxi_y = None

        self.width = None

    def set_data(self,x,y,update_ranges=True):
        self.lines['main'].set_data(x,y)
        if update_ranges:
            self.update_ranges(x,y)
        
    def update_ranges(self,x,y):
        self.mini_y = np.min(y)
        self.maxi_y = np.max(y)
        self.width = len(x)

    def get_lines(self):
        return [self.lines[k] for k in self.lines]

    def get_y_range(self):
        return self.mini_y,self.maxi_y

    def get_width(self):
        return self.width
