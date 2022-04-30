from plotter.plots.plot import Plot

class ActionPlot(Plot):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.title = 'Action Plot'
        self.xlabel = 'Iterations'
        self.ylabel = 'Action index'

    def plot_fn(self,*args,**kwargs):
        line = self.ax.scatter(*args,**kwargs)
        return line

    @property
    def targets(self):
        return ['Action']
