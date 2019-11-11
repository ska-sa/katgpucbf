import sys
import argparse
import math

import numpy as np
from bokeh.models import ColumnDataSource, Range1d
from bokeh.models.mappers import LinearColorMapper, LogColorMapper
from bokeh.plotting import curdoc, figure
from bokeh.layouts import gridplot
import colorcet

import backend


def make_figure(x_range, y_range):
    return figure(x_range=x_range, y_range=y_range,
                  tooltips=[("x", "$x"), ("y", "$y"),
                            ("mag", "@mag"), ("phase", "@phase")])


args = backend.parse_args(sys.argv[1:])

doc = curdoc()

shape = (args.acc_len, args.channels)
plots = []
x_range = Range1d(0, args.channels, bounds=(0, args.channels))
y_range = Range1d(0, args.acc_len, bounds=(0, args.acc_len))
for pol in range(2):
    source = ColumnDataSource(
        name=f'source{pol}',
        data={
            'mag': [np.zeros(shape)],
            'phase': [np.zeros(shape)],
            'x': [0],
            'y': [0],
            'dw': [args.channels],
            'dh': [args.channels]
        })

    pmag = make_figure(x_range, y_range)
    mapper = LogColorMapper(colorcet.fire, low=0.0, high=128.0 * math.sqrt(2))
    pmag.image(image='mag', x='x', y='y', dw='dw', dh='dh',
               color_mapper=mapper, source=source)

    pphase = make_figure(x_range, y_range)
    mapper = LinearColorMapper(colorcet.colorwheel, low=-math.pi, high=math.pi)
    pphase.image(image='phase', x='x', y='y', dw='dw', dh='dh',
                 color_mapper=mapper, source=source)
    plots.append([pmag, pphase])

grid = gridplot(plots, sizing_mode='stretch_both')

doc.add_root(grid)
