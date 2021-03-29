#!/usr/bin/env python3

import matplotlib.pyplot as plt

"""Set some plotting defaults"""
plotParams = {"font.size": 14,\
              "font.family": "serif",\
              "axes.titlesize": "medium",\
              "axes.labelsize": "small",\
              "axes.labelweight": "normal",\
              "axes.titleweight": "normal",\
              "patch.linewidth": 0.8,\
              "xtick.direction": "in",\
              "xtick.labelsize": "small",\
              "ytick.direction": "in",\
              "ytick.labelsize": "small",\
              "figure.titlesize": "medium",\
              "figure.titleweight": "normal",\
              "lines.linewidth": 1,\
              "lines.marker": "None",\
              "lines.markersize": 2,\
              "lines.markeredgewidth": 1,\
              "legend.edgecolor": "0",\
              "legend.framealpha": 1,\
              "legend.fontsize": "small"}

plt.rcParams.update(**plotParams)

