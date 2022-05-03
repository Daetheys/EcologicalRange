import matplotlib.pyplot as plt
import re
import numpy as np

class Plot:
    def __init__(self,ax):
        self.ax = ax

        self.title = ""
        self.xlabel = ""
        self.ylabel = ""

        self.maxi_y = 0
        self.mini_y = 0

        self.width = 1

        self.lines = {}

    def init_legend(self):
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)

    @property
    def flatten_lines(self):
        return [self.lines[k] for k in self.lines]

    def filter_labels(self,d):
        lbls = []
        for k in d:
            for t in self.targets:
                if re.match('^[a-zA-Z]+_[0-9]+/'+str(t)+'(?:_[0-9]+|)$',k):
                    lbls.append(k)
        return lbls
        
    @property
    def targets(self):
        return []

    def plot_fn(self,*args,**kwargs):
        line, = self.ax.plot(*args,**kwargs)
        return line

    def update_height(self,data):
        self.mini_y = min(self.mini_y,min(data))
        self.maxi_y = max(self.maxi_y,max(data))

    def get_height(self):
        r = self.maxi_y-self.mini_y
        mini = self.mini_y-r*0.1
        maxi = self.maxi_y+r*0.1
        if r == 0:
            mini = 0
            maxi = 1
        return mini,maxi

    def update_width(self,data):
        self.width = max(self.width,len(data))

    def get_width(self):
        return max(self.width,1)*1.1

    def show(self,d):
        for t in self.filter_labels(d):
            data = d[t]
            lbl = re.findall('^[a-zA-Z]+_[0-9]+/([a-zA-Z]+(?:_[0-9]+|))$',t)[0]
            try:
                if hasattr(self.lines[lbl],"set_data"):
                    self.lines[lbl].set_data(range(len(data)),data)
                elif hasattr(self.lines[lbl],"set_offsets"):
                    self.lines[lbl].set_offsets(np.c_[range(len(data)),data])
                else:
                    print(lbl,self.lines[lbl])
                    assert False #Unkown update function

                self.update_width(data)
                self.ax.set_xlim(0,self.get_width())

                self.update_height(data)
                mini,maxi = self.get_height()
                self.ax.set_ylim(mini,maxi)

            except KeyError:
                self.lines[lbl] = self.plot_fn(range(len(data)),data,label=lbl)
                if self.maxi_y is None:
                    self.maxi_y = max(data)
                if self.mini_y is None:
                    self.mini_y = min(data)
                self.ax.legend()
                
