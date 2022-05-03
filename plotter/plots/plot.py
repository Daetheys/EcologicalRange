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

    def plot_fn(self,x,y_mean,y_std,c=None,lw=1,label=None):
        line_mean, = self.ax.plot(x,y_mean,c=c,lw=lw,label=label)
        x = np.array(list(x))
        line_std = self.ax.fill_between(x,y_mean-y_std,y_mean+y_std,color=c)
        return line_mean,line_std

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
        filtered_data = {}
        for t in self.filter_labels(d):
            data = d[t]
            agent,lbl = re.findall('^([a-zA-Z]+_[0-9]+)/([a-zA-Z]+(?:_[0-9]+|))$',t)[0]
            try:
                filtered_data[lbl].append(np.array(data))
            except KeyError:
                filtered_data[lbl] = [np.array(data)]
        self.ax.collections.clear()
        for lbl in filtered_data:
            minlen = np.inf
            for l in filtered_data[lbl]:
                minlen = min(len(l),minlen)
            data = np.array([filtered_data[lbl][i][:minlen] for i in range(len(filtered_data[lbl]))])
            mean_data = np.mean(data,axis=0)
            std_data = np.var(data,axis=0)**0.5
            try:
                if hasattr(self.lines[lbl+'_mean'],"set_data"):
                    self.lines[lbl+'_mean'].set_data(range(len(mean_data)),mean_data)
                    self.lines[lbl+'_std'] = self.ax.fill_between(range(len(mean_data)),mean_data-std_data,mean_data+std_data)
                elif hasattr(self.lines[lbl],"set_offsets"):
                    self.lines[lbl].set_offsets(np.c_[range(len(data)),data])
                else:
                    print(lbl,self.lines[lbl])
                    assert False #Unkown update function

                self.update_width(mean_data)
                self.ax.set_xlim(0,self.get_width())

                self.update_height(mean_data)
                mini,maxi = self.get_height()
                self.ax.set_ylim(mini,maxi)

            except KeyError:
                self.lines[lbl+'_mean'],self.lines[lbl+'_std'] = self.plot_fn(range(len(mean_data)),mean_data,std_data,label=lbl)
                if self.maxi_y is None:
                    self.maxi_y = max(mean_data+std_data)
                if self.mini_y is None:
                    self.mini_y = min(mean_data-std_data)
                self.ax.legend()
                
