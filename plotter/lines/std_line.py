import matplotlib.pyplot as plt
from plotter.lines.line import Line
import numpy as np

def stdline(*args,**kwargs):
    return lambda lbl : StdLine(*args,label=lbl,**kwargs)

class StdLine(Line):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.lines['std'] = self.ax.fill_between([],[],[],color=self.color,alpha=0.2)
        
    def set_data(self,x,data,update_ranges=True):
        mean = np.mean(data,axis=-1)
        std = np.std(data,axis=-1)
        super().set_data(x,mean,update_ranges=False)
        self.ax.fill_between(x,mean-std,mean+std,color=self.color,alpha=0.2)
        if update_ranges:
            self.update_ranges(x,mean-std)
            self.update_ranges(x,mean+std)
