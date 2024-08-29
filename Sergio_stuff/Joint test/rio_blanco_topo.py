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
from landlab.grid.create_network import network_grid_from_raster
from landlab import imshow_grid
import landlab.plot.graph as graph
from landlab.components import ChannelProfiler, FlowAccumulator, FlowDirectorSteepest, DepressionFinderAndRouter
from landlab import imshow_grid
from landlab.io import read_esri_ascii

# import my DumbComponent
from landlab.components import BedRockAbrasionCoverEroder as BRACE
# and some tools
from mytools import Grid_geometry as geom
YEAR = 365.25 * 24 * 60 * 60
# %% -------------

# read ascii and store it as a raster object
filepath = "C:/Users/Sergio/Documents/GitHub/landlab_projects/Puerto Rico/Qgis_10m_resolution/rio_blanco_raster.asc"
rastergrid, topography = read_esri_ascii(filepath)
rastergrid.add_field("topographic__elevation", topography)
# plot spatially the topography
imshow_grid(rastergrid, rastergrid.at_node["topographic__elevation"], cmap='inferno_r')

# %% -------------
# watershed boundary
outletid = rastergrid.set_watershed_boundary_condition("topographic__elevation", return_outlet_id=True)
# %%
# flow direction
flow = FlowAccumulator(rastergrid, flow_director="D8")
flow.run_one_step()
# %%
# fill out sinks
df = DepressionFinderAndRouter(rastergrid)
df.map_depressions()
# %%
# get channels
profiler = ChannelProfiler(rastergrid,
                           minimum_channel_threshold=100,
                           main_channel_only=False)
profiler.run_one_step()

# %% ----------
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
BRACE._preset_fields(
    ngrid=ngrid,
    discharge=discharge,
    channel_width=channel_width,
    flood_intermittency=intermittency,
    sediment_grain_size=sediment_size,
    sed_capacity=initial_sed_capacity,
    macroroughness=macroroughness,
    mean_alluvium_thickness=initial_allu_thickness)
nety = BRACE(ngrid, flow_director,
             au=allu_smooth, su=slope_smooth)
# %%
# ploting channels
j, u = nety._find_joints()
channels = nety._get_channels()
# plotting with channel profiler didn't work
# ChannelProfiler(ngrid , channel_definition_field="discharge", minimum_channel_threshold=0)


# %%
# run one step
q_default = 0.000834
q_in = np.array([q_default, 3 * q_default])
nety.run_one_step(dt=YEAR / 1000, q_in=q_in)
xs = {}
for channel, nodes in channels.items():
    x = np.zeros_like(nodes)
    x[1:] = np.cumsum(nety.downstream_distance[nodes])[:-1]
    xs[channel] = x
plt.plot(xs[0])


# %%
# folder to save run results
folder_name = "Yfork"
filesname = "fork_test"
# laptop folder
"""savedir = path.Path("C:/Users/Paquito/Desktop/"
                    + "GitHub/Sharing/Nicole/runs/"
                    + folder_name)
"""
# lab pc folder
savedir = path.Path("C:/Users/Sergio/Documents/"
                    + "GitHub/Sharing/Nicole/runs/"
                    + folder_name)

# create folders
plotsdir = savedir / "plots"
datadir = savedir / "data"
os.makedirs(plotsdir, exist_ok=True)
os.makedirs(datadir, exist_ok=True)
# %%

# testing the get channels functionality

channels = nety._get_channels()


# %%

