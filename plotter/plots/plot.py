import matplotlib.pyplot as plt
import re

class Plot:
    def __init__(self,ax):
        self.ax = ax

        self.title = ""
        self.xlabel = ""
        self.ylabel = ""

        self.maxi_y = None
        self.mini_y = None

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

    def show(self,d):
        for t in self.filter_labels(d):
            data = d[t]
            lbl = re.findall('^[a-zA-Z]+_[0-9]+/([a-zA-Z]+(?:_[0-9]+|))$',t)[0]
            try:
                self.lines[lbl].set_data(range(len(data)),data)
                self.ax.set_xlim(0,len(data)*1.1)
                self.mini_y = min(self.mini_y,min(data))
                self.maxi_y = max(self.maxi_y,max(data))
                r = self.maxi_y-self.mini_y
                self.ax.set_ylim(self.mini_y-r*0.1,self.maxi_y+r*0.1)
            except KeyError:
                self.lines[lbl] = self.plot_fn(range(len(data)),data,label=lbl)
                if self.maxi_y is None:
                    self.maxi_y = max(data)
                if self.mini_y is None:
                    self.mini_y = min(data)
                self.ax.legend()
                
