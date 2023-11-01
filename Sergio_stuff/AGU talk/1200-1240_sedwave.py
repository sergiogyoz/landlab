# %%
# imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colorsmaps
import numpy as np
import pandas as pd
import math
import pathlib as path
import os

from landlab import RasterModelGrid
# from landlab import NetworkModelGrid
from landlab.components import FlowDirectorSteepest
from landlab.grid.create_network import network_grid_from_raster

# import my DumbComponent
from landlab.components import Componentcita as comp
# and some tools
from mytools import Model1D as m1d
YEAR = 365.25 * 24 * 60 * 60


# %%
# 40 year after steady state
context, records, sed_data = m1d.model1D(total_time=1240 * YEAR,
                                         record_time=2 * YEAR,
                                         total_length=20000,
                                         scale_of_high_feed=3,
                                         fraction_at_high_feed=0.25,
                                         cycle_period=40 * YEAR)
# %%
# folder to save run results
folder_name = "steady_sed_waves-2"
filesname = "1200y-1240y"
savedir = path.Path("C:/Users/Sergio/Documents/"
                    + "GitHub/Sharing/Nicole/runs/"
                    + folder_name)
# create folders
plotsdir = savedir / "plots"
datadir = savedir / "data"
os.makedirs(plotsdir)
os.makedirs(datadir)
# %%
m1d.plot_sed_graph(sed_data, filesname, plotsdir)
m1d.plot_1D_fields(context, records, filesname, savedir=plotsdir,
               from_time=1200 * YEAR, to_time=1240 * YEAR)
m1d.save_records_csv(records, datadir, filesname)
# %%
