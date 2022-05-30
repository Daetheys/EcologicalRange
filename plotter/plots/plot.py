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

        self.targets = [] #To define

    def init_legend(self):
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)

    @property
    def flatten_lines(self):
        out = []
        for _,l in self.lines.items():
            if hasattr(l,"get_lines"):
                out += l.get_lines()
            else:
                out.append(l)
        return out

    def filter_labels(self,d):
        lbls = []
        for k in d:
            for t in self.targets:
                if re.match('^[a-zA-Z]+_[0-9]+/'+str(t)+'(?:_[0-9]+|)$',k):
                    lbls.append(k)
        return lbls

    def plot_fn(self,x,y_mean,y_std,c=None,lw=1,label=None):
        line_mean, = self.ax.plot(x,y_mean,c=c,lw=lw,label=label)
        x = np.array(list(x))
        line_std = self.ax.fill_between(x,y_mean-y_std,y_mean+y_std,color=c)
        return line_mean,line_std

    def update_height(self,line):
        mini_y,maxi_y = line.get_y_range()
        if self.mini_y is None:
            self.mini_y = mini_y
        else:
            self.mini_y = min(self.mini_y,mini_y)
        if self.maxi_y is None:
            self.maxi_y = maxi_y
        else:
            self.maxi_y = max(self.maxi_y,maxi_y)

    def get_height(self):
        r = self.maxi_y-self.mini_y
        mini = self.mini_y-r*0.1
        maxi = self.maxi_y+r*0.1
        if r == 0:
            mini = 0
            maxi = 1
        return mini,maxi

    def update_width(self,line):
        width = line.get_width()
        if self.width is None:
            self.width = width
        else:
            self.width = max(self.width,width)

    def get_width(self):
        return max(self.width,1)*1.1

    def show(self,d):
        #Filter relevant data for this plot
        filtered_data = {}
        for t in self.filter_labels(d):
            data = d[t]
            agent,lbl = re.findall('^([a-zA-Z]+_[0-9]+)/([a-zA-Z]+(?:_[0-9]+|))$',t)[0]
            try:
                filtered_data[lbl].append(np.array(data))
            except KeyError:
                filtered_data[lbl] = [np.array(data)]
        #Plot data
        self.ax.collections.clear() #Clear collections
        for lbl in filtered_data:
            #Get maximal common length between each worker
            minlen = np.inf
            for l in filtered_data[lbl]:
                minlen = min(len(l),minlen)
            data = np.array([filtered_data[lbl][i][:minlen] for i in range(len(filtered_data[lbl]))])
            #print(data.shape)
            data = np.moveaxis(data,0,-1)
            #data = data.transpose()

            #Plot the data
            try:
                self.lines[lbl].set_data(range(len(data)),data)

            except KeyError:
                obj = re.findall('^([a-zA-Z]+)(?:_|)(?:[0-9]+|)$',lbl)[0]
                self.lines[lbl] = self.targets[obj](lbl)
                self.lines[lbl].set_data(range(len(data)),data)
                self.ax.legend()

            #Update the size of the figure
            self.update_width(self.lines[lbl])
            self.ax.set_xlim(0,self.get_width())
            
            self.update_height(self.lines[lbl])
            mini,maxi = self.get_height()
            self.ax.set_ylim(mini,maxi)
                
