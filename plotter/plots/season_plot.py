from plotter.plots.plot import Plot
import numpy as np


class SeasonPlot(Plot):
    def __init__(self,*args,season_length=10,colorspan='blue',alphaspan=0.2,**kwargs):
        print(args,kwargs)
        super().__init__(*args,**kwargs)
        self.season_length = season_length

        self.colorspan = colorspan
        self.alphaspan = alphaspan

        self.span_x_max = 0

    def show(self,d):
        mini,maxi = self.get_height()
        width = self.get_width()
        if width < self.span_x_max + self.season_length*2:
            self.ax.axvspan(self.span_x_max,self.span_x_max+self.season_length,alpha=self.alphaspan,facecolor=self.colorspan)
            self.span_x_max += self.season_length*2
        super().show(d)