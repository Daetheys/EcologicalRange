from plotter.plots.season_plot import SeasonPlot
import re

class RangeAgentPlot(SeasonPlot):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.title = 'RangeAgent Plot'
        self.xlabel = 'Iterations'
        self.ylabel = 'Min/Max Reward'

    def filter_labels(self,d):
        lbls = []
        for k in d:
            for t in self.targets:
                if re.match('^[a-zA-Z]+_[0-9]+/'+str(t)+'(?:_[0-9]+|)$',k):
                    lbls.append(k)
        return lbls

    @property
    def targets(self):
        return ['AgentMini','AgentMaxi']+['EnvMini','EnvMaxi']
