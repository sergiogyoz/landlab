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
from landlab.components import BedRockAbrassionCoverEroder as BRACE
# and some tools
from mytools import Model1D as m1d
YEAR = 365.25 * 24 * 60 * 60


# %%
# 40 year after steady state
context, records, sed_data = m1d.model1D(total_time=1200 * YEAR,
                                         record_time=100 * YEAR,
                                         total_length=20000)
# %%
# folder to save run results
folder_name = "1200y-100y"
filesname = "1200y_record_100y"
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
m1d.plot_1D_fields(context, records, filesname, savedir=plotsdir)
m1d.save_records_csv(records, datadir, filesname)
# %%
