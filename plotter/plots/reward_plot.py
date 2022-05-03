from plotter.plots.season_plot import SeasonPlot

class RewardPlot(SeasonPlot):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.title = 'Reward Plot'
        self.xlabel = 'Iterations'
        self.ylabel = 'Reward'

    @property
    def targets(self):
        return ['Reward','EV']
