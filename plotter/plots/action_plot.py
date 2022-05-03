from plotter.plots.season_plot import SeasonPlot

class ActionPlot(SeasonPlot):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.title = 'Action Plot'
        self.xlabel = 'Iterations'
        self.ylabel = 'Action index'

    def plot_fn(self,x,y,*args,**kwargs):
        line = self.ax.scatter(x,y,*args,**kwargs)
        return line

    @property
    def targets(self):
        return ['Action']
