from plotter.plots.season_plot import SeasonPlot
from plotter.lines.std_line import stdline
import re

class RangeAgentPlot(SeasonPlot):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.title = 'RangeAgent Plot'
        self.xlabel = 'Iterations'
        self.ylabel = 'Min/Max Reward'

        self.targets = {'AgentMini':stdline(self.ax,color='blue'),
                        'AgentMaxi':stdline(self.ax,color='red'),
                        'EnvMini':stdline(self.ax,color='orange'),
                        'EnvMaxi':stdline(self.ax,color='green')}

    def filter_labels(self,d):
        lbls = []
        for k in d:
            for t in self.targets:
                if re.match('^[a-zA-Z]+_[0-9]+/'+str(t)+'(?:_[0-9]+|)$',k):
                    lbls.append(k)
        return lbls
