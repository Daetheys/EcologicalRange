from plotter.plots.season_plot import SeasonPlot
from plotter.lines.scatter import scatter

class ActionProbPlot(SeasonPlot):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.title = 'ActionProb Plot'
        self.xlabel = 'Iterations'
        self.ylabel = 'Action index'

        self.targets ={'ActionProbs':scatter(self.ax,cmap='Greys')}
        #{'PVal':scatter(self.ax,cmap='Greys'),
        #                'QVal':scatter(self.ax,cmap='Greys')}
