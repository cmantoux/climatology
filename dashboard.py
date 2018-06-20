from ipywidgets import VBox, HBox, HTML, Layout
from bqplot import Axis, OrdinalScale, LinearScale, Lines
from bqplot.pyplot import Figure
import numpy as np

class Fig(Figure):
    def __init__(self, model, key):
        param = model.params[key]
        if type(param.value)==np.ndarray and key != 'T13':
            self.n = len(param.value)
        else:
            self.n = 1
        os = LinearScale()
        ls = LinearScale()
        ax_x = Axis(scale=os, grid_lines='solid', label='Iterations')
        ax_y = Axis(scale=ls, orientation='vertical', tick_format='0.2f', grid_lines='solid', label=key)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.lines = []
        for k in range(self.n):
            self.lines.append(Lines(x=np.array([]), y=np.array([]), scales={'x': os, 'y': ls}, colors = [colors[k]], stroke_width=2, display_legend=True, labels=['{}_{}'.format(key,k)]))

        super().__init__(marks=self.lines, axes=[ax_x, ax_y], title=key, legend_location='bottom-right')
        

class Dashboard(VBox):
    def __init__(self, model):
        self.dic_figures = {}
        self.labels = {}
        for key in model.params.keys():
            self.dic_figures[key] = Fig(model, key)
            self.labels[key] = HTML(layout = Layout(width='200px', height='200px', padding='20px'))
        super().__init__([HBox([self.dic_figures[figure], self.labels[figure]]) for figure in self.dic_figures])
