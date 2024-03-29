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
from landlab import NetworkModelGrid
from landlab.components import FlowDirectorSteepest
from landlab.grid.create_network import network_grid_from_raster
from landlab import imshow_grid
import landlab.plot.graph as graph

# import my DumbComponent
from landlab.components import BedRockAbrassionCoverEroder as BRACE
# and some tools
from mytools import Grid_geometry as geom
YEAR = 365.25 * 24 * 60 * 60

# %%
total_lenght = 2000
reach_lenght = 200
initial_slope = 0.004
discharge = 300
intermittency = 0.05
channel_width = 100
sediment_size = 0.02
initial_sed_capacity = 0
macroroughness = 1
initial_allu_thickness = 0.5
allu_smooth = 0.8
slope_smooth = 0.2
shape = (5, (5 + 1) // 2)
# %%
geo = geom(shape)
xs, ys, links, steep = geo.Yfork()
x_of_node = [x * reach_lenght for x in xs]
y_of_node = [y * reach_lenght for y in ys]

ngrid = NetworkModelGrid((y_of_node, x_of_node), links)
graph.plot_graph(ngrid, at="node,link", with_id=True)
# %%
ngrid.add_field("topographic__elevation", steep)
flow_director = FlowDirectorSteepest(ngrid)
flow_director.run_one_step()
# initial values and parameters of the network
BRACE.BedRockAbrassionCoverEroder._preset_fields(
    ngrid=ngrid,
    discharge=discharge,
    channel_width=channel_width,
    flood_intermittency=intermittency,
    sediment_grain_size=sediment_size,
    sed_capacity=initial_sed_capacity,
    macroroughness=macroroughness,
    mean_alluvium_thickness=initial_allu_thickness)
nety = BRACE.BedRockAbrassionCoverEroder(ngrid, flow_director,
                          au=allu_smooth, su=slope_smooth)
# %%
nety.run_one_step(dt=YEAR / 1000, q_in=0.000834)
"""See channel profiler to figure out how to get the network later"""
j, u = nety._find_joints()

# %%
# folder to save run results
folder_name = "Yfork"
filesname = "fork_test"
savedir = path.Path("C:/Users/Paquito/Desktop/"
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

