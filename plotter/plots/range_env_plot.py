from plotter.plots.plot import Plot

class RangeEnvPlot(Plot):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.title = 'RangeEnvironment Plot'
        self.xlabel = 'Iterations'
        self.ylabel = 'Min/Max Reward'

    @property
    def targets(self):
        return ['EnvMini','EnvMaxi']
