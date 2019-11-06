import sys
import argparse
import math

import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.models.mappers import LinearColorMapper, LogColorMapper
from bokeh.plotting import curdoc, figure
from bokeh.layouts import gridplot
import colorcet

import backend


def make_figure(args):
    return figure(x_range=[0, args.channels], y_range=[0, args.acc_len],
                  tooltips=[("x", "$x"), ("y", "$y"),
                            ("mag", "@mag"), ("phase", "@phase")])


args = backend.parse_args(sys.argv[1:])

doc = curdoc()

shape = (args.acc_len, args.channels)
source = ColumnDataSource(
    name='source',
    data={
        'mag': [np.zeros(shape)],
        'phase': [np.zeros(shape)],
        'x': [0],
        'y': [0],
        'dw': [args.channels],
        'dh': [args.channels]
    })

pmag = make_figure(args)
mapper = LogColorMapper(colorcet.fire, low=0.0, high=128.0 * math.sqrt(2))
pmag.image(image='mag', x='x', y='y', dw='dw', dh='dh',
           color_mapper=mapper, source=source)

pphase = make_figure(args)
mapper = LinearColorMapper(colorcet.colorwheel, low=-math.pi, high=math.pi)
pphase.image(image='phase', x='x', y='y', dw='dw', dh='dh',
             color_mapper=mapper, source=source)

grid = gridplot([[pmag, pphase]], sizing_mode='stretch_both')

doc.add_root(grid)
