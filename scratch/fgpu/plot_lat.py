#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

timestamps = np.loadtxt('timestamps.txt')
high = 0
lat = []
lat_ts = []
for ts in timestamps:
    if ts > high:
        high = ts
    else:
        lat.append(high - ts)
        lat_ts.append(ts)
plt.plot(np.array(lat_ts) / 1712e6, np.array(lat) / 1712e6, '.')
plt.show()
