import sys
import argparse

import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.plotting import curdoc, figure

import backend

args = backend.parse_args(sys.argv[1:])

doc = curdoc()

shape = (args.channels, args.acc_len)
source = ColumnDataSource(
    name='source',
    data={
        'mag': [np.zeros(shape)],
        'phase': [np.zeros(shape)],
        'x': [0],
        'y': [0],
        'dw': [args.channels],
        'dh': [args.channels],
        'palette': ['Spectral11']
    })
fig = figure(x_range=[0, args.channels], y_range=[0, args.acc_len])
fig.image(image='mag', x='x', y='y', dw='dw', dh='dh',
          palette='Spectral11', source=source)

doc.add_root(fig)
