from plotter.plots.season_plot import SeasonPlot

class RangeEnvPlot(SeasonPlot):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.title = 'RangeEnvironment Plot'
        self.xlabel = 'Iterations'
        self.ylabel = 'Min/Max Reward'

    @property
    def targets(self):
        return ['EnvMini','EnvMaxi']
