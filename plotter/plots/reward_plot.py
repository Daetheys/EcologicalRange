from plotter.plots.season_plot import SeasonPlot
from plotter.lines.std_line import stdline

class RewardPlot(SeasonPlot):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.title = 'Reward Plot'
        self.xlabel = 'Iterations'
        self.ylabel = 'Reward'

        self.targets = {'Reward':stdline(self.ax,color='green'),
                        'EV':stdline(self.ax,color='orange'),
                        'EnvMini':stdline(self.ax,color='blue'),
                        'EnvMaxi':stdline(self.ax,color='purple')}

